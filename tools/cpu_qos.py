"""Diagnostic HighQoS for a caller-owned Popen handle; never opens/enumerates PIDs.

Only EXECUTION_SPEED is controlled. Timer resolution and other bits are preserved.
See https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
"""
import ctypes
from ctypes import wintypes
import time

PROCESS_POWER_THROTTLING = 4
VERSION = 1
EXECUTION_SPEED = 1


class PowerThrottlingState(ctypes.Structure):
    _fields_ = [('Version', wintypes.DWORD), ('ControlMask', wintypes.DWORD),
                ('StateMask', wintypes.DWORD)]


def _api():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    for name in ('GetProcessInformation', 'SetProcessInformation'):
        function = getattr(api, name)
        function.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        function.restype = wintypes.BOOL
    return api


def _values(state):
    return {'version': state.Version, 'control_mask': state.ControlMask, 'state_mask': state.StateMask}


def _interpret(state):
    if not state.ControlMask & EXECUTION_SPEED:
        return 'system_managed_execution_speed_not_proof_of_EcoQoS'
    return 'explicit_EcoQoS' if state.StateMask & EXECUTION_SPEED else 'explicit_HighQoS'


def apply_high_qos(proc):
    """Use only proc._handle after the caller has assigned this Popen to its Job.

    An integer PID/handle argument is rejected. This helper never calls OpenProcess,
    changes priority/affinity, or touches a power plan. All times use perf_counter.
    """
    started = time.perf_counter()
    report = {'requested': True, 'applied_successfully': False, 'before': None, 'after': None,
              'desired': None, 'before_interpretation': None, 'after_interpretation': None,
              'no_op_already_high': False, 'setter_called': False, 'setter_passed': None,
              'setter_seconds': None, 'setter_started_seconds': None, 'setter_finished_seconds': None,
              'operation_seconds': None, 'errors': [],
              'handle_scope': 'Only supplied Popen._handle, after owned Job assignment; no PID lookup'}
    def error(operation, code=None, message=None):
        report['errors'].append({'operation': operation, 'win32_error': code, 'message': message})
    try:
        if isinstance(proc, (int, str, bytes)) or not hasattr(proc, '_handle'):
            raise ValueError('Expected the caller-owned Popen object, not a PID/handle argument')
        handle = int(proc._handle)
        if handle <= 0:
            raise ValueError('Invalid caller-owned Popen handle')
        api = _api()
        def read(label):
            state = PowerThrottlingState(VERSION, 0, 0)
            if not api.GetProcessInformation(handle, PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)):
                error(label, ctypes.get_last_error())
                return None
            if state.Version != VERSION:
                error(label, message='Unexpected power-throttling version')
                return None
            return state
        before = read('GetProcessInformation.before')
        if before is None:
            return report
        report['before'] = _values(before)
        report['before_interpretation'] = _interpret(before)
        desired = PowerThrottlingState(VERSION, before.ControlMask | EXECUTION_SPEED,
                                      before.StateMask & ~EXECUTION_SPEED)
        report['desired'] = _values(desired)
        report['no_op_already_high'] = _values(before) == _values(desired)
        if not report['no_op_already_high']:
            tick = time.perf_counter()
            report['setter_started_seconds'] = tick - started
            report['setter_called'] = True
            passed = api.SetProcessInformation(handle, PROCESS_POWER_THROTTLING,
                                               ctypes.byref(desired), ctypes.sizeof(desired))
            code = ctypes.get_last_error() if not passed else None
            finished = time.perf_counter()
            report['setter_seconds'] = finished - tick
            report['setter_finished_seconds'] = finished - started
            report['setter_passed'] = bool(passed)
            if not passed:
                error('SetProcessInformation', code)
        after = read('GetProcessInformation.after')
        if after is not None:
            report['after'] = _values(after)
            report['after_interpretation'] = _interpret(after)
            if _values(after) != _values(desired):
                error('Verify preserved masks', message='Readback differs from requested masks')
        report['applied_successfully'] = after is not None and not report['errors'] and (
            report['no_op_already_high'] or report['setter_passed'] is True)
    except Exception as failure:
        error('HighQoS diagnostic', message=f'{type(failure).__name__}: {failure}')
    finally:
        report['operation_seconds'] = time.perf_counter() - started
    return report
