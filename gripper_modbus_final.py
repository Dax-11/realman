# -*- coding: utf-8 -*-
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Robotic_Arm.rm_robot_interface import *


class Reg:
    MOTOR_ENABLE           = 256
    POS_MODE_SELECT        = 257
    TARGET_POS_HIGH        = 258
    TARGET_POS_LOW         = 259
    TARGET_SPEED           = 260
    TARGET_FORCE           = 261
    RAMP_UP                = 262
    RAMP_DOWN              = 263
    MOTION_TRIGGER         = 264
    ALARM_RESET            = 272
    POWERON_ZERO_SWITCH    = 275
    POWERON_ENABLE_SWITCH  = 276
    SAVE_ALL_PARAMS        = 282

    MOTION_STATE           = 1025
    ALARM_A                = 1026
    ALARM_B_DROP           = 1027
    REAL_POS_HIGH           = 1048
    REAL_POS_LOW             = 1049
    REAL_SPEED               = 1050
    REAL_CURRENT             = 1051


class MotionState:
    POSITION_REACHED   = 0x0001
    SPEED_REACHED       = 0x0002
    ZERO_SPEED_REACHED  = 0x0004
    FORCE_REACHED        = 0x0008
    READY                 = 0x0010
    MOTOR_ENABLED         = 0x8000

    @staticmethod
    def parse(raw: int) -> dict:
        return {
            'position_reached': bool(raw & MotionState.POSITION_REACHED),
            'speed_reached': bool(raw & MotionState.SPEED_REACHED),
            'zero_speed_reached': bool(raw & MotionState.ZERO_SPEED_REACHED),
            'force_reached': bool(raw & MotionState.FORCE_REACHED),
            'ready': bool(raw & MotionState.READY),
            'motor_enabled': bool(raw & MotionState.MOTOR_ENABLED),
            'raw': raw,
        }


class PositionMode:
    ABSOLUTE = 0
    RELATIVE = 1
    VELOCITY = 2
    TORQUE = 3


class GripperModbusController:
    POS_MIN = 0
    POS_MAX = 9000

    def __init__(self, arm: 'RoboticArm', device_id: int = 1, baudrate: int = 115200):
        self.arm = arm
        self.device_id = device_id
        self.baudrate = baudrate
        self._legacy_modbus = False

    def switch_to_modbus_rtu_mode(self) -> bool:
        if hasattr(self.arm, "rm_set_rm_plus_mode"):
            plus_ret = self.arm.rm_set_rm_plus_mode(0)
            print(f"[信息] 关闭末端生态协议 rm_set_rm_plus_mode(0) 返回码: {plus_ret}")
            time.sleep(0.5)

        ret = self.arm.rm_set_tool_rs485_mode(mode=0, baudrate=self.baudrate)
        print(f"[信息] 切换工具端RS485为RTU主站 rm_set_tool_rs485_mode(0, {self.baudrate}) 返回码: {ret}")
        if ret != 0:
            print(f"[警告] 四代工具端RS485接口切换失败，返回码: {ret}")
            if hasattr(self.arm, "rm_set_modbus_mode"):
                legacy_ret = self.arm.rm_set_modbus_mode(1, self.baudrate, 10)
                print(f"[信息] 尝试旧接口 rm_set_modbus_mode(1, {self.baudrate}, 10) 返回码: {legacy_ret}")
                if legacy_ret == 0:
                    self._legacy_modbus = True
                    time.sleep(0.3)
                    return True
            print("[错误] 切换Modbus RTU主站模式失败")
            return False
        time.sleep(0.3)
        self._legacy_modbus = False
        try:
            ret2, info = self.arm.rm_get_tool_rs485_mode_v4()
            print(f"[信息] 查询工具端RS485模式返回码: {ret2}, 当前RS485模式: {info}")
        except Exception as exc:
            print(f"[警告] 查询工具端RS485模式失败: {exc}")
        return ret == 0

    def switch_to_rm_plus_mode(self, baudrate: int = 115200) -> bool:
        ret = self.arm.rm_set_rm_plus_mode(baudrate)
        return ret == 0

    def write_single(self, addr: int, value: int) -> int:
        if self._legacy_modbus:
            param = rm_peripheral_read_write_params_t(
                port=1, address=addr, device=self.device_id
            )
            return self.arm.rm_write_single_register(param, value)

        param = rm_modbus_rtu_write_params_t(
            address=addr, device=self.device_id, type=1,
            num=1, data=[value]
        )
        ret = self.arm.rm_write_modbus_rtu_registers(param)
        if ret == -4 and hasattr(self.arm, "rm_write_single_register"):
            self._legacy_modbus = True
            return self.write_single(addr, value)
        return ret

    def write_multi(self, addr: int, values: list) -> int:
        if self._legacy_modbus:
            data_bytes = []
            for value in values:
                value &= 0xFFFF
                data_bytes.extend([(value >> 8) & 0xFF, value & 0xFF])
            param = rm_peripheral_read_write_params_t(
                port=1, address=addr, device=self.device_id, num=len(values)
            )
            return self.arm.rm_write_registers(param, data_bytes)

        param = rm_modbus_rtu_write_params_t(
            address=addr, device=self.device_id, type=1,
            num=len(values), data=values
        )
        ret = self.arm.rm_write_modbus_rtu_registers(param)
        if ret == -4 and hasattr(self.arm, "rm_write_registers"):
            self._legacy_modbus = True
            return self.write_multi(addr, values)
        return ret

    def read_holding(self, addr: int, count: int = 1) -> list:
        if self._legacy_modbus:
            if count == 1:
                param = rm_peripheral_read_write_params_t(
                    port=1, address=addr, device=self.device_id
                )
                ret, value = self.arm.rm_read_holding_registers(param)
                if ret != 0:
                    print(f"[警告] 旧接口读寄存器 addr={addr} 失败，返回码: {ret}")
                    return []
                return [value]

            if count <= 2:
                data = []
                for offset in range(count):
                    part = self.read_holding(addr + offset, 1)
                    if not part:
                        return []
                    data.extend(part)
                return data

            param = rm_peripheral_read_write_params_t(
                port=1, address=addr, device=self.device_id, num=count
            )
            ret, data_bytes = self.arm.rm_read_multiple_holding_registers(param)
            if ret != 0:
                print(f"[警告] 旧接口读多个寄存器 addr={addr} count={count} 失败，返回码: {ret}")
                return []
            regs = []
            for i in range(count):
                hi = data_bytes[2 * i] & 0xFF
                lo = data_bytes[2 * i + 1] & 0xFF
                regs.append((hi << 8) | lo)
            return regs

        param = rm_modbus_rtu_read_params_t(
            address=addr, device=self.device_id, type=1, num=count
        )
        ret, data = self.arm.rm_read_modbus_rtu_holding_registers(param)
        if ret == -4 and hasattr(self.arm, "rm_read_holding_registers"):
            self._legacy_modbus = True
            return self.read_holding(addr, count)
        if ret != 0:
            print(f"[警告] 读寄存器 addr={addr} 失败，返回码: {ret}")
            return []
        return data

    def enable(self) -> int:
        return self.write_single(Reg.MOTOR_ENABLE, 1)

    def disable(self) -> int:
        return self.write_single(Reg.MOTOR_ENABLE, 0)

    def reset_alarm(self) -> int:
        return self.write_single(Reg.ALARM_RESET, 1)

    def save_params_to_flash(self) -> int:
        return self.write_single(Reg.SAVE_ALL_PARAMS, 1)

    def set_power_on_auto_zero(self, enable: bool) -> int:
        return self.write_single(Reg.POWERON_ZERO_SWITCH, 1 if enable else 0)

    def set_power_on_auto_enable(self, enable: bool) -> int:
        return self.write_single(Reg.POWERON_ENABLE_SWITCH, 1 if enable else 0)

    def move(self, position: int, speed: int = 50, force: int = 50,
              ramp_up: int = 1000, ramp_down: int = 1000,
              mode: int = PositionMode.ABSOLUTE, trigger: bool = True) -> int:
        position = max(self.POS_MIN, min(position, self.POS_MAX))
        pos_high = (position >> 16) & 0xFFFF
        pos_low = position & 0xFFFF

        steps = [
            (Reg.POS_MODE_SELECT, mode),
            (Reg.TARGET_POS_HIGH, pos_high),
            (Reg.TARGET_POS_LOW, pos_low),
            (Reg.TARGET_SPEED, speed),
            (Reg.TARGET_FORCE, force),
            (Reg.RAMP_UP, ramp_up),
            (Reg.RAMP_DOWN, ramp_down),
        ]
        for addr, val in steps:
            ret = self.write_single(addr, val)
            if ret != 0:
                print(f"[错误] 写入地址{addr}失败，返回码: {ret}")
                return ret

        if trigger:
            return self.write_single(Reg.MOTION_TRIGGER, 1)
        return 0

    def grip(self, position: int = 9000, speed: int = 50, force: int = 30,
              ramp_up: int = 1000, ramp_down: int = 1000) -> int:
        return self.move(position=position, speed=speed, force=force,
                          ramp_up=ramp_up, ramp_down=ramp_down,
                          mode=PositionMode.ABSOLUTE, trigger=True)

    def release(self, speed: int = 50, force: int = 100,
                ramp_up: int = 1000, ramp_down: int = 1000) -> int:
        return self.move(position=0, speed=speed, force=force,
                          ramp_up=ramp_up, ramp_down=ramp_down,
                          mode=PositionMode.ABSOLUTE, trigger=True)

    def get_motion_state(self) -> dict:
        data = self.read_holding(Reg.MOTION_STATE, 1)
        if not data:
            return {}
        return MotionState.parse(data[0])

    def get_alarm_a(self) -> int:
        data = self.read_holding(Reg.ALARM_A, 1)
        return data[0] if data else None

    def is_workpiece_dropped(self) -> bool:
        data = self.read_holding(Reg.ALARM_B_DROP, 1)
        return bool(data and data[0] == 1000)

    def get_real_position(self) -> int:
        data = self.read_holding(Reg.REAL_POS_HIGH, 2)
        if len(data) < 2:
            return None
        hi, lo = data[0], data[1]
        combined = (hi << 16) | lo
        if combined & 0x80000000:
            combined -= 0x100000000
        return combined

    def get_real_speed(self) -> int:
        data = self.read_holding(Reg.REAL_SPEED, 1)
        return data[0] if data else None

    def get_real_current(self) -> int:
        data = self.read_holding(Reg.REAL_CURRENT, 1)
        return data[0] if data else None

    def get_full_status(self) -> dict:
        return {
            'motion_state': self.get_motion_state(),
            'alarm_a': self.get_alarm_a(),
            'dropped': self.is_workpiece_dropped(),
            'position': self.get_real_position(),
            'speed': self.get_real_speed(),
            'current': self.get_real_current(),
        }

    def wait_for_motion_complete(self, timeout: float = 5.0,
                                  poll_interval: float = 0.02,
                                  verbose: bool = False) -> dict:
        start = time.time()
        force_reached_first = False
        position_reached_first = False

        while time.time() - start < timeout:
            state = self.get_motion_state()
            if verbose:
                pos = self.get_real_position()
                print(f"  t={time.time()-start:.2f}s 位置={pos} 状态={state}")

            if not state:
                time.sleep(poll_interval)
                continue

            if state.get('force_reached') and not position_reached_first:
                force_reached_first = True
            if state.get('position_reached') and not force_reached_first:
                position_reached_first = True

            if state.get('zero_speed_reached') and state.get('ready'):
                elapsed = time.time() - start
                final_pos = self.get_real_position()
                return {
                    'completed': True,
                    'grip_success': force_reached_first and not position_reached_first,
                    'final_state': state,
                    'final_position': final_pos,
                    'elapsed': elapsed,
                }

            time.sleep(poll_interval)

        return {
            'completed': False,
            'grip_success': False,
            'final_state': self.get_motion_state(),
            'final_position': self.get_real_position(),
            'elapsed': timeout,
        }

    def grip_and_wait(self, position: int = 9000, speed: int = 50, force: int = 30,
                       timeout: float = 5.0, verbose: bool = False) -> dict:
        ret = self.grip(position=position, speed=speed, force=force)
        if ret != 0:
            print(f"[错误] 抓取指令发送失败，返回码: {ret}")
            return {'completed': False, 'grip_success': False, 'send_error': ret}
        return self.wait_for_motion_complete(timeout=timeout, verbose=verbose)

    def release_and_wait(self, speed: int = 50, timeout: float = 5.0,
                          verbose: bool = False) -> dict:
        ret = self.release(speed=speed)
        if ret != 0:
            print(f"[错误] 松开指令发送失败，返回码: {ret}")
            return {'completed': False, 'send_error': ret}
        return self.wait_for_motion_complete(timeout=timeout, verbose=verbose)

    def diagnose(self, duration: float = 3.0, interval: float = 0.1) -> None:
        print(f"[诊断模式] 持续 {duration} 秒打印状态...")
        start = time.time()
        while time.time() - start < duration:
            status = self.get_full_status()
            print(f"  t={time.time()-start:.2f}s {status}")
            time.sleep(interval)


if __name__ == "__main__":
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
    print(f"连接句柄: {handle.id}")

    if handle.id == -1:
        print("连接失败，请检查IP地址和网络连接")
        sys.exit(1)

    gripper = GripperModbusController(arm, device_id=1, baudrate=115200)

    print("\n=== 切换协议模式 ===")
    if not gripper.switch_to_modbus_rtu_mode():
        print("协议切换失败，退出")
        arm.rm_delete_robot_arm()
        sys.exit(1)

    print("\n=== 电机使能 ===")
    ret = gripper.enable()
    print(f"使能结果: {ret}")
    time.sleep(0.5)

    gripper.reset_alarm()

    print("\n=== 全开夹爪 ===")
    result = gripper.release_and_wait(speed=50, verbose=True)
    print(f"结果: {result}")

    input("\n放好物体后按回车，开始力控抓取（force=30，较温和）...")
    print("\n=== 力控抓取 ===")
    result = gripper.grip_and_wait(position=9000, speed=50, force=30, verbose=True)
    print(f"\n最终结果: {result}")

    if result.get('grip_success'):
        print(f">>> 抓取成功！力矩优先到达，说明夹到了物体")
        print(f">>> 最终位置: {result['final_position']}")
    elif result.get('completed'):
        print(f">>> 位置优先到达，夹爪走到底，可能未夹到物体")
    else:
        print(">>> 未在超时内完成，运行诊断...")
        gripper.diagnose(duration=3)

    print("\n=== 完整状态 ===")
    print(gripper.get_full_status())

    print("\n=== 松开 ===")
    gripper.release_and_wait(speed=50, verbose=True)

    arm.rm_delete_robot_arm()
    print("\n已断开连接")
