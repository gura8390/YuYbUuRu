// Private b9090 Vulkan cache helper. No exported ABI and no extra loader imports.
#pragma once
#include <vulkan/vulkan_core.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#endif

namespace ggml_vk_cache {

class state {
#ifdef _WIN32
    static constexpr std::size_t limit = 64 * 1024 * 1024;
    static constexpr std::array<std::uint8_t, 8> magic{'Y','Y','U','R','V','K','C','1'};
    struct file_handle {
        HANDLE value = INVALID_HANDLE_VALUE;
        explicit file_handle(HANDLE handle) : value(handle) {}
        file_handle(const file_handle&) = delete;
        file_handle& operator=(const file_handle&) = delete;
        ~file_handle() { close(); }
        void close() { if (value != INVALID_HANDLE_VALUE) { CloseHandle(value); value = INVALID_HANDLE_VALUE; } }
    };
    VkDevice device = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    PFN_vkCreatePipelineCache create = nullptr;
    PFN_vkDestroyPipelineCache release = nullptr;
    PFN_vkGetPipelineCacheData get_data = nullptr;
    std::wstring filename;
    std::mutex save_mutex;
    std::atomic<bool> warned{false};
    std::atomic<bool> bypassed{false};

    static std::uint32_t read_u32(const std::uint8_t* p) {
        return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
               (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    }
    static void write_u32(std::uint8_t* p, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(value >> (i * 8));
    }
    static std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
        static constexpr auto table = [] {
            std::array<std::uint32_t, 256> result{};
            for (std::uint32_t i = 0; i < 256; ++i) {
                auto value = i;
                for (int bit = 0; bit < 8; ++bit) value = (value >> 1) ^ ((value & 1) ? 0xedb88320u : 0);
                result[i] = value;
            }
            return result;
        }();
        std::uint32_t value = 0xffffffffu;
        for (std::size_t i = 0; i < size; ++i) value = table[(value ^ data[i]) & 255] ^ (value >> 8);
        return value ^ 0xffffffffu;
    }
    bool valid_header(const std::uint8_t* data, std::size_t size) const {
        return size >= 32 && size <= limit && read_u32(data) == 32 &&
            read_u32(data + 4) == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
            read_u32(data + 8) == properties.vendorID && read_u32(data + 12) == properties.deviceID &&
            std::memcmp(data + 16, properties.pipelineCacheUUID, VK_UUID_SIZE) == 0;
    }
    void warning(const char* operation) {
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_WARN("ggml_vulkan: pipeline cache %s failed; inference remains available\n", operation);
        }
    }
    std::vector<std::uint8_t> read_initial() {
        file_handle file(CreateFileW(filename.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                      nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
        if (file.value == INVALID_HANDLE_VALUE) return {};
        LARGE_INTEGER length{};
        if (!GetFileSizeEx(file.value, &length) || length.QuadPart < 48 || length.QuadPart > limit + 16) {
            GGML_LOG_DEBUG("ggml_vulkan: pipeline cache size rejected\n"); return {};
        }
        std::vector<std::uint8_t> envelope(static_cast<std::size_t>(length.QuadPart));
        DWORD read = 0;
        if (!ReadFile(file.value, envelope.data(), static_cast<DWORD>(envelope.size()), &read, nullptr) ||
                read != envelope.size() || !std::equal(magic.begin(), magic.end(), envelope.begin()) ||
                read_u32(envelope.data() + 8) != envelope.size() - 16 ||
                read_u32(envelope.data() + 12) != crc32(envelope.data() + 16, envelope.size() - 16) ||
                !valid_header(envelope.data() + 16, envelope.size() - 16)) {
            GGML_LOG_DEBUG("ggml_vulkan: pipeline cache data rejected\n"); return {};
        }
        envelope.erase(envelope.begin(), envelope.begin() + 16);
        GGML_LOG_INFO("ggml_vulkan: pipeline cache loaded %zu bytes\n", envelope.size());
        return envelope;
    }

public:
    // All Vulkan entry points come from the existing instance dispatch table.
    void initialize(VkDevice logical, const VkPhysicalDeviceProperties& physical,
            PFN_vkCreatePipelineCache create_fn, PFN_vkDestroyPipelineCache destroy_fn,
            PFN_vkGetPipelineCacheData data_fn) noexcept {
        const DWORD count = GetEnvironmentVariableW(L"GGML_VK_PIPELINE_CACHE_DIR", nullptr, 0);
        if (!count || count > 30000 || !create_fn || !destroy_fn || !data_fn) return;
        try {
            device = logical; properties = physical;
            create = create_fn; release = destroy_fn; get_data = data_fn;
            std::wstring directory(count, L'\0');
            const DWORD used = GetEnvironmentVariableW(L"GGML_VK_PIPELINE_CACHE_DIR", directory.data(), count);
            if (!used || used >= count) return;
            directory.resize(used);
            std::error_code error;
            const auto folder = std::filesystem::absolute(std::filesystem::path(directory), error);
            if (error) { warning("path"); return; }
            std::filesystem::create_directories(folder, error);
            if (error) { warning("directory"); return; }
            std::ostringstream key;
            key << "yyur-v1-" << std::hex << properties.vendorID << '-' << properties.deviceID << '-' << properties.driverVersion << '-';
            constexpr char hex[] = "0123456789abcdef";
            for (auto value : properties.pipelineCacheUUID) key << hex[value >> 4] << hex[value & 15];
            key << ".cache";
            filename = (folder / key.str()).wstring();
            const auto initial = read_initial();
            VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            // Default flags retain Vulkan's internal cache synchronization.
            info.initialDataSize = initial.size();
            info.pInitialData = initial.empty() ? nullptr : initial.data();
            auto result = create(device, &info, nullptr, &cache);
            if (result != VK_SUCCESS && !initial.empty()) {
                cache = VK_NULL_HANDLE; info.initialDataSize = 0; info.pInitialData = nullptr;
                result = create(device, &info, nullptr, &cache);
            }
            if (result != VK_SUCCESS) { cache = VK_NULL_HANDLE; warning("create"); }
        } catch (...) { warning("initialize"); }
    }

    VkPipelineCache handle() const noexcept { return bypassed.load(std::memory_order_acquire) ? VK_NULL_HANDLE : cache; }
    void bypass_after_create_failure() noexcept {
        // Keep the object alive: another compile may already be using it.
        // Preserve the disk file since a resource failure need not indicate corrupt data.
        if (!bypassed.exchange(true, std::memory_order_acq_rel)) {
            GGML_LOG_WARN("ggml_vulkan: cached pipeline creation failed; cache disabled for this device, retrying without cache\n");
        }
    }

    // Called only after successful pipeline creation: worker termination may skip native destructors.
    // The entire bounded read/CRC/write/flush/rename cost remains in the caller's prefill interval.
    void save() noexcept {
        if (!cache || bypassed.load(std::memory_order_acquire)) return;
        try {
            std::lock_guard<std::mutex> guard(save_mutex);
            if (bypassed.load(std::memory_order_acquire)) return;
            std::size_t size = 0;
            auto result = get_data(device, cache, &size, nullptr);
            if (result != VK_SUCCESS || size < 32 || size > limit) { warning("data size"); return; }
            std::vector<std::uint8_t> envelope(size + 16);
            result = get_data(device, cache, &size, envelope.data() + 16);
            // A concurrent compile may grow the cache. Skip this snapshot on VK_INCOMPLETE;
            // the next successful compile saves another complete snapshot.
            if (result != VK_SUCCESS || size > envelope.size() - 16 ||
                    !valid_header(envelope.data() + 16, size)) { warning("snapshot"); return; }
            envelope.resize(size + 16);
            std::copy(magic.begin(), magic.end(), envelope.begin());
            write_u32(envelope.data() + 8, static_cast<std::uint32_t>(size));
            write_u32(envelope.data() + 12, crc32(envelope.data() + 16, size));
            LARGE_INTEGER tick{}; QueryPerformanceCounter(&tick);
            const auto temporary = filename + L"." + std::to_wstring(GetCurrentProcessId()) + L"." +
                std::to_wstring(GetCurrentThreadId()) + L"." + std::to_wstring(tick.QuadPart) + L".tmp";
            file_handle file(CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr));
            if (file.value == INVALID_HANDLE_VALUE) { warning("temporary file"); return; }
            DWORD wrote = 0;
            const bool complete = WriteFile(file.value, envelope.data(), static_cast<DWORD>(envelope.size()), &wrote, nullptr) &&
                                  wrote == envelope.size() && FlushFileBuffers(file.value);
            file.close();
            if (bypassed.load(std::memory_order_acquire)) { DeleteFileW(temporary.c_str()); return; }
            const bool renamed = complete && MoveFileExW(temporary.c_str(), filename.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
            if (!renamed) { DeleteFileW(temporary.c_str()); warning("atomic save"); return; }
            GGML_LOG_DEBUG("ggml_vulkan: pipeline cache saved %zu bytes\n", size);
        } catch (...) { warning("save"); }
    }

    void destroy() noexcept {
        if (cache) { release(device, cache, nullptr); cache = VK_NULL_HANDLE; }
    }
#else
public:
    void initialize(VkDevice, const VkPhysicalDeviceProperties&, PFN_vkCreatePipelineCache,
            PFN_vkDestroyPipelineCache, PFN_vkGetPipelineCacheData) noexcept {}
    VkPipelineCache handle() const noexcept { return VK_NULL_HANDLE; }
    void bypass_after_create_failure() noexcept {}
    void save() noexcept {}
    void destroy() noexcept {}
#endif
};

} // namespace ggml_vk_cache
