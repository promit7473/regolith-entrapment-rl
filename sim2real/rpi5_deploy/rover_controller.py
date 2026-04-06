"""
Phase 3 — Onboard rover controller for Raspberry Pi 5.

Runs a 10 Hz control loop:
  1. Read wheel encoders → wheel velocities
  2. Read MPU6050 IMU → linear acceleration
  3. Build 12D observation (matches training env)
  4. Run sinkage detector (ONNX) → state classification
  5. If entrapped: run recovery policy (ONNX) → wheel commands
  6. Command motors via GPIO PWM

Hardware assumptions:
  - 4× DC motors with quadrature encoders (GPIO pins defined below)
  - MPU6050 IMU via I2C (bus 1, addr 0x68)
  - L298N dual H-bridge ×2 for motor drive
  - Python: RPi.GPIO, smbus2, onnxruntime, numpy (all pip-installable)

Usage (on RPi5):
    python rover_controller.py \
        --policy_onnx  recovery_policy.onnx \
        --detector_onnx sinkage_detector.onnx \
        --run_time 300

Simulator stub: if GPIO is not available (dev machine), the script
runs with a software stub so logic can be tested offline.
"""

from __future__ import annotations

import argparse
import collections
import time
import os
import sys
import numpy as np

# ── Hardware constants ─────────────────────────────────────────────────────
WHEEL_RADIUS       = 0.098   # m
ENCODER_PPR        = 1440    # pulses per revolution (encoder spec)
MAX_WHEEL_VEL      = 10.0    # rad/s (matches training env)
CTRL_HZ            = 10      # Hz
CTRL_DT            = 1.0 / CTRL_HZ
SEQ_LEN            = 50      # steps in detector window
ENTRAPPED_HOLD     = 5       # consecutive "entrapped" predictions before engaging policy

# GPIO pin assignments (BCM numbering)
# Motor 1 = front-left, 2 = front-right, 3 = rear-left, 4 = rear-right
MOTOR_IN1  = [5,  6,  13, 19]
MOTOR_IN2  = [11, 12, 16, 20]
MOTOR_ENA  = [18,  8, 25, 24]    # PWM-capable pins
ENC_A      = [17, 27,  22, 10]
ENC_B      = [4,   3,  23,  9]

IMU_BUS    = 1
IMU_ADDR   = 0x68
GRAVITY    = 9.81


# ── GPIO + IMU abstraction (stub if hardware unavailable) ──────────────────

class _GPIOStub:
    BCM = OUT = IN = 0
    def setmode(self, *a): pass
    def setup(self, *a, **k): pass
    def output(self, *a): pass
    def cleanup(self): pass
    class PWM:
        def __init__(self, *a): pass
        def start(self, *a): pass
        def ChangeDutyCycle(self, *a): pass
        def stop(self): pass

class _SMBusStub:
    def __init__(self, *a): pass
    def read_byte_data(self, *a): return 0
    def write_byte_data(self, *a): pass

try:
    import RPi.GPIO as GPIO
    _gpio_available = True
except ImportError:
    GPIO = _GPIOStub()
    _gpio_available = False

try:
    import smbus2
    _i2c_available = True
except ImportError:
    smbus2 = None
    _i2c_available = False


class EncoderReader:
    """Quadrature encoder reader using GPIO interrupts."""

    def __init__(self):
        self._counts = [0] * 4
        if not _gpio_available:
            return
        for i in range(4):
            GPIO.setup(ENC_A[i], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(ENC_B[i], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            # Closure captures index
            def make_cb(idx):
                def cb(ch):
                    b = GPIO.input(ENC_B[idx])
                    self._counts[idx] += 1 if b else -1
                return cb
            GPIO.add_event_detect(ENC_A[i], GPIO.RISING, callback=make_cb(i))

    def get_velocities_rad_s(self, dt: float) -> np.ndarray:
        """Return wheel angular velocities in rad/s and reset counters."""
        counts = list(self._counts)
        self._counts = [0] * 4
        vel = [(c / ENCODER_PPR) * (2 * np.pi) / dt for c in counts]
        return np.array(vel, dtype=np.float32)


class MPU6050:
    """Minimal MPU6050 driver over I2C."""
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B

    def __init__(self):
        if not _i2c_available:
            self._bus = None
            return
        self._bus = smbus2.SMBus(IMU_BUS)
        self._bus.write_byte_data(IMU_ADDR, self.PWR_MGMT_1, 0)  # wake up

    def read_accel_ms2(self) -> np.ndarray:
        """Return [ax, ay, az] in m/s²."""
        if self._bus is None:
            return np.zeros(3, dtype=np.float32)
        raw = []
        for reg in [self.ACCEL_XOUT_H, self.ACCEL_XOUT_H+2, self.ACCEL_XOUT_H+4]:
            hi = self._bus.read_byte_data(IMU_ADDR, reg)
            lo = self._bus.read_byte_data(IMU_ADDR, reg + 1)
            val = (hi << 8) | lo
            if val > 32767:
                val -= 65536
            raw.append(val)
        scale = GRAVITY / 16384.0   # ±2g range
        return np.array(raw, dtype=np.float32) * scale


class MotorDriver:
    """PWM motor driver via L298N."""

    def __init__(self):
        self._pwm = []
        if not _gpio_available:
            return
        for i in range(4):
            GPIO.setup(MOTOR_IN1[i], GPIO.OUT)
            GPIO.setup(MOTOR_IN2[i], GPIO.OUT)
            GPIO.setup(MOTOR_ENA[i], GPIO.OUT)
            p = GPIO.PWM(MOTOR_ENA[i], 1000)   # 1 kHz PWM
            p.start(0)
            self._pwm.append(p)

    def set_wheel_velocities(self, vels_normalised: np.ndarray):
        """
        vels_normalised: array of 4 values in [-1, 1].
        Positive = forward rotation.
        """
        if not _gpio_available:
            return
        for i, v in enumerate(vels_normalised):
            duty = min(abs(float(v)) * 100, 100)
            if v >= 0:
                GPIO.output(MOTOR_IN1[i], GPIO.HIGH)
                GPIO.output(MOTOR_IN2[i], GPIO.LOW)
            else:
                GPIO.output(MOTOR_IN1[i], GPIO.LOW)
                GPIO.output(MOTOR_IN2[i], GPIO.HIGH)
            self._pwm[i].ChangeDutyCycle(duty)

    def stop(self):
        if not _gpio_available:
            return
        for p in self._pwm:
            p.ChangeDutyCycle(0)
            p.stop()


# ── Main controller ────────────────────────────────────────────────────────

class RoverController:
    """
    10 Hz closed-loop controller integrating sinkage detection and recovery policy.
    """

    STATE_NORMAL    = 0
    STATE_SINKING   = 1
    STATE_ENTRAPPED = 2

    def __init__(self, policy_onnx: str, detector_onnx: str):
        import onnxruntime as ort
        self._policy   = ort.InferenceSession(policy_onnx,
                             providers=["CPUExecutionProvider"])
        self._detector = ort.InferenceSession(detector_onnx,
                             providers=["CPUExecutionProvider"])

        self._enc    = EncoderReader()
        self._imu    = MPU6050()
        self._motors = MotorDriver()

        # Rolling sequence buffer for detector: (SEQ_LEN, 11)
        self._seq_buf = collections.deque(
            [np.zeros(11, dtype=np.float32)] * SEQ_LEN, maxlen=SEQ_LEN
        )

        self._entrapped_count = 0
        self._state           = self.STATE_NORMAL

        # Running mean/std for observation normalisation (initialised online)
        self._obs_mean = np.zeros(12, dtype=np.float32)
        self._obs_m2   = np.ones(12, dtype=np.float32)
        self._obs_n    = 0

    # ── Observation builder ───────────────────────────────────────────────

    def _build_obs(self, wheel_vel: np.ndarray, imu_acc: np.ndarray,
                   slip: np.ndarray) -> np.ndarray:
        """Build 12D obs matching the training environment."""
        grav_z = np.array([-1.0], dtype=np.float32)   # assume upright by default
        obs = np.concatenate([
            wheel_vel / MAX_WHEEL_VEL,   # 4D normalised
            slip,                        # 4D slip ratio
            imu_acc / GRAVITY,           # 3D normalised acceleration
            grav_z,                      # 1D tilt proxy
        ])
        return obs.astype(np.float32)

    def _build_detector_feat(self, wheel_vel: np.ndarray,
                              wheel_torque: np.ndarray,
                              imu_acc: np.ndarray) -> np.ndarray:
        """Build 11D feature vector for detector."""
        return np.concatenate([
            wheel_vel / MAX_WHEEL_VEL,
            wheel_torque / 20.0,
            imu_acc / GRAVITY,
        ]).astype(np.float32)

    # ── Online obs normalisation ─────────────────────────────────────────

    def _update_normalise(self, obs: np.ndarray) -> np.ndarray:
        self._obs_n += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_n
        self._obs_m2   += delta * (obs - self._obs_mean)
        if self._obs_n > 10:
            std = np.sqrt(self._obs_m2 / self._obs_n).clip(1e-4)
            return (obs - self._obs_mean) / std
        return obs

    # ── Inference ────────────────────────────────────────────────────────

    def _run_detector(self) -> int:
        seq = np.stack(list(self._seq_buf), axis=0)[np.newaxis]   # (1, 50, 11)
        logits = self._detector.run(None, {"sequence": seq})[0]   # (1, 3)
        return int(np.argmax(logits, axis=-1)[0])

    def _run_policy(self, obs: np.ndarray) -> np.ndarray:
        obs_in = obs[np.newaxis].astype(np.float32)
        action = self._policy.run(None, {"obs": obs_in})[0][0]    # (4,)
        return np.clip(action, -1.0, 1.0)

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self, run_time_s: float = 300.0):
        GPIO.setmode(GPIO.BCM)
        print(f"\n[RoverController] Starting — run_time={run_time_s}s  hz={CTRL_HZ}")
        print(f"  GPIO: {'available' if _gpio_available else 'STUB (dev mode)'}")
        print(f"  I2C:  {'available' if _i2c_available else 'STUB'}\n")

        t_start = time.time()
        step    = 0

        try:
            while (time.time() - t_start) < run_time_s:
                t0 = time.time()

                # ── Sense ──────────────────────────────────────────────
                wheel_vel    = self._enc.get_velocities_rad_s(CTRL_DT)
                imu_acc      = self._imu.read_accel_ms2()
                wheel_torque = np.zeros(4, dtype=np.float32)  # no torque sensor; use zeros

                # Slip ratio
                body_vel = np.mean(wheel_vel) * WHEEL_RADIUS     # rough approximation
                eps      = 0.01
                denom    = np.maximum(np.abs(wheel_vel * WHEEL_RADIUS),
                                      max(abs(body_vel), eps))
                slip     = np.clip((wheel_vel * WHEEL_RADIUS - body_vel) / denom, -1, 1)

                # ── Detector ───────────────────────────────────────────
                det_feat = self._build_detector_feat(wheel_vel, wheel_torque, imu_acc)
                self._seq_buf.append(det_feat)

                if step % 5 == 0:    # run detector at 2 Hz (every 5 ctrl steps)
                    pred = self._run_detector()
                    if pred == self.STATE_ENTRAPPED:
                        self._entrapped_count += 1
                    else:
                        self._entrapped_count = max(0, self._entrapped_count - 1)

                    if self._entrapped_count >= ENTRAPPED_HOLD:
                        self._state = self.STATE_ENTRAPPED
                    elif pred == self.STATE_SINKING:
                        self._state = self.STATE_SINKING
                    else:
                        self._state = self.STATE_NORMAL

                # ── Act ────────────────────────────────────────────────
                obs = self._build_obs(wheel_vel, imu_acc, slip)
                obs = self._update_normalise(obs)

                if self._state == self.STATE_ENTRAPPED:
                    action = self._run_policy(obs)
                    print(f"  [{step:5d}] ENTRAPPED → policy action: {np.round(action, 2)}")
                elif self._state == self.STATE_SINKING:
                    # Rocking maneuver as fallback
                    sign   = 1 if (step // 10) % 2 == 0 else -1
                    action = np.full(4, sign, dtype=np.float32)
                    print(f"  [{step:5d}] SINKING   → rocking: {sign:+d}")
                else:
                    # Normal: forward drive
                    action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

                self._motors.set_wheel_velocities(action)

                step += 1
                elapsed = time.time() - t0
                sleep   = max(0.0, CTRL_DT - elapsed)
                time.sleep(sleep)

        except KeyboardInterrupt:
            print("\n[RoverController] Interrupted by user.")
        finally:
            self._motors.stop()
            GPIO.cleanup()
            print(f"[RoverController] Stopped after {step} steps "
                  f"({step * CTRL_DT:.1f} s).")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RPi5 Rover Controller")
    parser.add_argument("--policy_onnx",   type=str, required=True)
    parser.add_argument("--detector_onnx", type=str, required=True)
    parser.add_argument("--run_time",      type=float, default=300.0,
                        help="Total run time in seconds")
    args = parser.parse_args()

    if not os.path.isfile(args.policy_onnx):
        raise FileNotFoundError(f"Policy ONNX not found: {args.policy_onnx}")
    if not os.path.isfile(args.detector_onnx):
        raise FileNotFoundError(f"Detector ONNX not found: {args.detector_onnx}")

    ctrl = RoverController(args.policy_onnx, args.detector_onnx)
    ctrl.run(args.run_time)


if __name__ == "__main__":
    main()
