# FR10 teleoperation

`sim_fr10.py` runs Isaac Lab. `teleop_fr10.py` receives headset input, solves
inverse kinematics, and controls the simulator or real FR10. `camera_fr10.py`
supplies the Orbbec Gemini 2 color stream for the real robot. USB capture runs
in a separate process from robot control.

Run commands from this directory:

```bash
cd /home/shashank/Projects/xr_teleoperate/fr10_teleop
```

## Environments

Use the existing `env_isaaclab` environment for simulation and `xr` for control.
Both already have their core dependencies on this workstation. The camera SDK
is an additional dependency.

For a new control environment, the solver needs **Pinocchio with CasADi
bindings** and CasADi itself. The installed `xr` environment uses Python 3.10,
Pinocchio 3.2.0, and CasADi 3.6.7. A matching setup is:

```bash
conda create -n fr10_xr -c conda-forge python=3.10 pinocchio=3.2.0 casadi=3.6.7
conda activate fr10_xr
python -m pip install -r requirements.txt
python -m pip install -e ../teleop/televuer
python -c "import pinocchio; from pinocchio import casadi; print(pinocchio.__version__)"
```

Keep Isaac Lab in its own environment; do not install control requirements over
its managed packages. Install the optional camera SDK in `xr` or a separate
compatible Python environment:

```bash
conda activate xr
python -m pip install -r requirements-camera.txt
```

The current official SDK v2 supports the ordinary **Gemini 2**. Its pip
distribution is `pyorbbecsdk2`; the import remains `pyorbbecsdk`. Check the
[official setup instructions](https://github.com/orbbec/pyorbbecsdk) for Linux USB
permissions and the [release support matrix](https://github.com/orbbec/pyorbbecsdk/releases)
for compatible firmware. Gemini 2 XL is a different model with legacy SDK
requirements. Firmware and USB rules are not installed by this project.

## Try control without hardware

```bash
conda activate xr
python teleop_fr10.py --robot mock --input scripted --auto-start --duration 5 --no-images
```

This runs synthetic hand motion through the real IK, clutch state machine and
joint filtering, then stops after five seconds. It does not open a simulation
window. Scripted input and automatic start are limited to `mock` and `sim`.

## Isaac Lab simulation

Terminal 1:

```bash
conda activate env_isaaclab
python sim_fr10.py
```

Terminal 2, using a headset:

```bash
conda activate xr
python teleop_fr10.py --robot sim --record
```

Or use synthetic input without a headset:

```bash
conda activate xr
python teleop_fr10.py --robot sim --input scripted --auto-start --duration 10 --no-images
```

The simulator holds its initial pose until commands arrive. Its camera is
virtual; the physical Gemini 2 is used on the real setup. Only one publisher can
bind the image port at a time: stop the physical camera server before starting
the simulator with its camera enabled.

For a bounded simulator check without camera capture:

```bash
python sim_fr10.py --headless --no-camera --max-steps 200
```

Isaac Sim runtime/rendering still needs verification with a working NVIDIA
driver. The unit tests below do not exercise Isaac Sim.

## Gemini 2 camera and real FR10

Verify camera discovery and capture before connecting to the robot:

```bash
conda activate xr
python camera_fr10.py --list-devices
python camera_fr10.py --config config/orbbec_gemini2.yaml --preview
```

Leave the server running. `--preview` is optional and requires a desktop.
Use `--max-frames 100` for bounded capture. With multiple cameras attached, select
one with `--serial YOUR_SERIAL` or `--set orbbec.serial=YOUR_SERIAL`. Unsupported
resolution/rate combinations produce an error listing available color profiles.

In a second terminal:

```bash
conda activate xr
python teleop_fr10.py --robot real --config config/orbbec_gemini2.yaml --record
```

Defaults use IP `192.168.58.2` and the SDK directory in `config/default.yaml`.
Set the actual IP and folder containing the vendor's `Robot.py` or `Robot.*.so`
if different:

```bash
python teleop_fr10.py --robot real --config config/orbbec_gemini2.yaml --record \
  --robot-ip 192.168.58.2 --sdk-path /absolute/path/to/fairino
```

Connecting invokes the existing FAIRINO backend: it can switch to automatic
mode, enable the arm, and activate its gripper. The launcher waits for `r`
before streaming arm set-points. The default gripper index is `2`; match the
pendant configuration, or use `--set gripper.enabled=false`. Controller errors
are reported; `--reset-errors` requests error clearing on connection.

The Gemini 2 preset shows a **monocular** color image in the headset's `ego`
window. Configure HTTPS certificates as described in
[TeleVuer's README](../teleop/televuer/README.md), then open
`https://<PC_IP>:8012?ws=wss://<PC_IP>:8012` in the headset browser and enter VR.
The headset and PC must be reachable on the network.

If capture runs on another PC, set `image.host` to its address in the
teleoperation command. The JPEG port defaults to `55555`; the simulator uses
additional command/state ports `5557`/`5556`.

This implementation streams and records **color images only**. Raw depth,
intrinsics, color/depth alignment and hand-eye calibration are not stored in
the JPEG path. A wrist-mounted camera alone does not establish a transform
from depth pixels to the robot base. Relative headset teleoperation does not
require that transform; depth-based perception would.

## Controls and recording

Press keys in the teleoperation terminal, without Enter:

| Key | Action |
| --- | --- |
| `r` | Start streaming or recover from a fault |
| `Space` | **Stop the arm immediately** (program keeps running; `r` resumes) |
| `h` | Move to the configured home pose |
| `c` | Start/stop an episode when launched with `--record` |
| `o` | Open the gripper now (real robot) |
| `g` | Close the gripper now (real robot); your hand takes over once it matches or moves clearly |
| `x` | Reset the simulator scene |
| `+` / `-` | Increase/decrease speed scale |
| `q` or Ctrl+C | Stop and close the program |

The right controller drives the arm by default. Release then squeeze its grip
button to engage the clutch; release to hold position. The trigger controls the
gripper, A toggles recording, and B requests home. For hand tracking use
`--set xr.input_mode=hand`: press `r`, hold your right hand palm down, fingers forward, and make
a **fist with your left hand** to make the arm follow (open the left hand to stop following).
On the real FR10 the gripper uses the calls of the standalone `gripper.py` (`ActGripper(2, 1)`, then
`MoveGripper(2, position, 100, 67, 2000, ...)`) on its own controller connection, the same code as
`tools/pinch_gripper.py`, so a slow gripper command never waits on the arm. The gripper **follows how far
you pinch**: thumb-index 80 mm or more is open, 20 mm or less is fully closed, and a half pinch (50 mm)
stops the gripper halfway and keeps it there; closing the right hand into a fist closes it fully.
A new position is sent only when it changes by 5 %. When a session starts, the gripper is driven once to the
opening your hand shows (a gripper that is really closed but reports itself as open would otherwise never be
opened), and `ActGripper` is called before that first move and after a refused move. Press `o` in the
terminal to open the gripper at any time. Tune the range with
`xr.gripper_pinch_open_m` / `xr.gripper_pinch_closed_m`. The gripper runs in its own thread. `ActGripper` takes seconds on a gripper that is not active yet (it
calibrates), so it is called once per session and the moves after it return at once (`block=1`); calling it
before every move, or waiting for every move as `gripper.py` does, makes calls time out
(`fairino.gripper.activate_before_move` and `fairino.gripper.wait_for_move` restore that, and
`fairino.gripper.activation_timeout_s` is its time limit). The thread wakes at once on a key or a pinch, logs
every command with the controller's answer, and the `ready | ...` / `tracking | ...` lines show its state
(`gripper thread running ...; last command ...`, or `gripper thread BLOCKED in a controller call ...` if the
controller does not answer). It sends plain XML-RPC calls with a timeout on its own socket;
`fairino.gripper.separate_sdk_connection: true` opens a second vendor-SDK connection instead, which the
controller may refuse. `fairino.gripper.mode: open_close` gives only open or closed.

**Smooth following of a lagging arm.** A real arm in servo mode trails its command by a few tenths of a
second. The command may run ahead of the measured arm by `control.soft_command_lead_deg` (10 degrees) with no
effect at all; from there to `control.max_command_lead_deg` (22 degrees) its velocity limit falls smoothly to
zero (braking stays at full strength); at the maximum it waits, and `control.stall_timeout_s` turns a wait that
does not end into a fault. A hard, low limit (it was 6 degrees) makes the motion stop and go: measured with a
0.4 s lag it tripled the acceleration and left the command standing still for up to 14 % of the time, and the
tests keep that from coming back. The `tracking |` line shows `arm lag slowed N %, held N % of ticks`; both
should stay near 0 in normal motion.

**Arm set-points over UDP.** The arm's `ServoJ` commands go over the vendor SDK's UDP pass-through
(`fairino.servo_transport: udp`, the default): fire and forget, like the DDS commands of the original project,
and not queued behind the controller's XML-RPC replies, so gripper and set-up calls cannot stall the arm. The
log says which transport is used (`arm set-points (ServoJ) go over UDP`). It needs controller firmware that
supports it; if the arm never moves, the fault message says so, and `--set fairino.servo_transport=xmlrpc`
switches back. `config/default.yaml` is loaded by the program and overrides the code default, so keep it in
step (the tests check that).

**Gripper process.** The gripper is built like the Inspire hand controller of the original project: a process of
its own (`fr10/gripper_process.py`), fed through a shared value (`robot.set_gripper` only writes it, so the
arm's loop never waits), running a fixed 20 Hz loop that sends few commands and one exact final command when the
fingers stop. It activates the gripper once when the program connects, and `r` waits for that to finish so the
slow activation never overlaps the arm's commands. `fairino.gripper.use_process: false` goes back to a thread
inside the arm's process. One terminal is enough:

```bash
./scripts/start_teleop_real.sh
```

**Gripper and arm share one queue in the controller.** Measured on the FR10 controller (V3.9.4) through the
real-time state packet: `MoveGripper` runs in the same motion queue as the arm's `ServoJ` set-points
(`mc_queue_len`). While a gripper move runs (about 1.2 s for a full stroke) the arm stands still, whatever is
streamed. If set-points keep coming during the move, they pile up behind it (the queue grew to 160 = 1.3 s).
Since they then arrive exactly as fast as they run, the arm lags 1.3 s for the rest of the session, and every
later `MoveGripper` waits for a queue that never empties (the held moves all run when the teleoperation ends).
UDP or XML-RPC, `ServoMoveStart`, `block` 0 or 1, a second program or thread: none of these changes that,
because it happens inside the controller. With `fairino.gripper.pause_arm_for_moves: true` (the default), each
gripper move during teleoperation goes like this:

1. The gripper process asks the arm's servo thread to pause. The arm brakes smoothly to rest and stops sending.
2. The gripper move is sent and taken at once, because the queue is empty.
3. The arm stays silent until the state packet reports the move done (`gripper_motiondone`) and the queue empty.
4. The arm follows your hand again, with no added lag.

On the robot: the arm paused 1.4 s per move, and its lag behind the set-point was 78–88 ms before and after
every move. To keep pauses rare, the gripper gets a new position only once your fingers have settled (within
`still_band_percent`) or are fully open or closed, with at least `pause_cooldown_s` between moves: one pause
per grasp and one per release. The tracking line shows `ARM PAUSED while the gripper moves` and the number of
pauses. `pause_timeout_s` is a safety limit: the arm goes on anyway if the gripper process never answers.

A latency guard (`fairino.max_queue_points`) watches the controller queue as well:
- More than that many points queued: nothing is sent until the queue is back to one.
- At least 2 points queued for 10 ticks in a row: one tick is skipped (that happens after timing jitter or
  clock drift).

Normally the queue holds 0–1 points and the guard does nothing; if it acts, the tracking line says so. Only
the gripper process can pause the arm, so the thread path (`use_process: false`) and the separate pinch tool
below get just one gripper move through during teleoperation.

**Gripper in its own program.** To run the arm and the gripper as two separate programs, start the
teleoperation without gripper control (it owns the headset and shares your hand data on `127.0.0.1:5560`),
then the pinch tool in a second terminal, which needs no headset connection of its own:

```bash
python teleop_fr10.py --robot real --no-gripper --set xr.input_mode=hand --set control.speed_scale=0.2
python tools/pinch_gripper.py --from-teleop            # pinch = closed, release = open
python tools/pinch_gripper.py --from-teleop --proportional   # a half pinch stops the gripper halfway
```

The tool talks to the controller with plain XML-RPC (`--connection sdk` uses the vendor SDK like `gripper.py`,
which a second client next to the teleoperation may not tolerate). It activates the gripper once when it
starts (this can take several seconds, so start it before pressing `r` in the teleoperation) and then sends
moves that return at once; `--gripper-py-style` activates before and waits for every move like `gripper.py`. With `--no-gripper` the teleoperation does
not touch the gripper, so its `o` / `g` keys do nothing.

The **right hand's closure sets the gripper**: an open hand is an open gripper, a half-closed
hand closes it halfway, a fist closes it fully. After pressing `r` the gripper holds its opening
until your hand matches it, so starting never opens a gripper that is holding a part; if tracking
of the right hand is lost the last opening is kept. The left fist is the clutch. Other options
(`--set xr.gripper_source=...`): `both_hands_pinch` (a pinch of either hand, other fingers
straight, toggles open/closed), `other_hand_pinch` (left hand only) and `driving_hand_pinch`
(thumb-index distance of the right hand). Tune the range with `xr.gripper_fist_open_ratio` and
`xr.gripper_fist_closed_ratio`.

**Ready pose.** Pressing `r` first drives the arm smoothly to the home (ready) pose, then waits
for the clutch; if the arm is already there it just becomes ready. The move is a joint-space
straight line, at `control.home_speed_deg_s` and the speed scale, so clear the path first.
Use `--no-home-on-start` to hold the arm where it is instead. `h` (or the B button) sends the
arm home at any time. Motion is relative to the hand pose at the moment the clutch closes, so
the arm never chases a hand that is far away.

**Gripper orientation follows your hand.** Hold the hand with the palm facing down and the
fingers pointing forward: the gripper points down. Turning the hand turns the gripper the
same way. The clutch only engages once your hand roughly matches the gripper (within
`mapping.engage_tolerance_deg`, 35 degrees by default), so the arm never swings on engage;
if it does not engage, the terminal says how far off the hand is. If palm-down does not give
a downward gripper with your controller, tune `mapping.hand_offset_rpy_deg`
(roll, pitch, yaw in degrees), or use the old behaviour that copies only the rotation since
the clutch closed with `--set mapping.orientation_mode=relative`.

Adjust camera-relative motion on hardware with
`--set mapping.yaw_offset_deg=ANGLE`. Automatic heading comes from the virtual
camera in simulation and defaults to zero on hardware.

Episodes are written under `data/<task>/episode_NNNN/` with `meta.json`,
`steps.jsonl`, and optional JPEG images. `--record` enables recording; press
`c` or A after starting to begin an episode. Images are included when a recent
frame is available. `--no-images` disables image reception and switches the
headset to pass-through. `--duration SECONDS` bounds control-loop runtime.

All programs accept `--config FILE` and repeated `--set section.key=value`.
Partial YAML files overlay built-in defaults; unknown keys are rejected.
Run each script with `--help` for full options.

## Verification

```bash
conda activate xr
python -m unittest discover -s tests -v
```

Tests cover kinematics, IK, control modes, startup/shutdown, recording, camera
buffer conversion/profile selection and the actual local ZeroMQ simulator
link. Camera tests use a fake SDK and synthetic buffers. They do not verify
USB hardware, real robot motion, headset networking or Isaac rendering.
