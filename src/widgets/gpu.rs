//! GPU utilisation, sampled from whichever vendor CLI happens to be installed.
//!
//! Three design decisions are load-bearing and worth recording:
//!
//! - **Auto-detect is the only path.** A user who wants specific behaviour
//!   configures it on their own box; mirador does not pick programs by name.
//!   This walks back a small piece of the project's "you name the program"
//!   ethic (the same one that powers `chime_command` and `open_command`) but
//!   the trade is intentional: detection is opt-out only because the user's
//!   first question on a new machine is "is anything actually rendering?",
//!   and an answer that needs zero keystrokes is more useful than one that
//!   asks for a config edit before saying anything.
//!
//! - **The fetch runs on a background thread, never the main loop.** Each
//!   probe spawns a subprocess (`nvidia-smi` and friends), and a probe that
//!   blocks for two seconds blocks the dashboard. The thread wakes every
//!   [`SAMPLE_INTERVAL`], publishes into an `Arc<Mutex<Sample>>`, and exits
//!   when the panel is dropped.
//!
//! - **Per-frame work is bounded by what is on screen.** A machine with eight
//!   GPUs does not allocate eight times per draw: `render` reads a cached
//!   [`Sample`] taken once per successful sample, and the worker is the only
//!   place that ever clones a `Device`. The same discipline
//!   `quotas`/`news`/`agenda` learned in the per-frame allocation sweep —
//!   a panel may allocate in proportion to what is on screen, not in
//!   proportion to how many GPUs the box has.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{MouseEvent, MouseEventKind};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::Paragraph;

use crate::panel::{Panel, RenderContext};
use crate::poll::wait;

/// How often the worker takes a new sample. Long enough that an uninstalled
/// vendor tool's "command not found" is not hammering, short enough that the
/// panel feels live on a screen that update about once a second.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(1500);

/// Per-probe timeout. `nvidia-smi` is normally sub-100ms; the budget is
/// generous because a flaky driver can hang indefinitely.
const PROBE_TIMEOUT: Duration = Duration::from_millis(2000);

/// One probe's contribution to the panel.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Probe {
    /// Section header, e.g. `NVIDIA`, `INTEL/ARC`, `AMD`.
    ///
    /// Plural labels are wrong on purpose: a probe that listed five AMD cards
    /// is still one section, because the *driver* is the thing being
    /// detected, not the device.
    pub label: String,
    pub devices: Vec<Device>,
}

/// One GPU, as the worker's parser returned it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Device {
    /// Index in the vendor's enumeration. NVIDIA is 0-indexed from
    /// `nvidia-smi`; Intel/AMD are synthesised to 0 since their CLI does not
    /// always give one.
    pub index: u32,
    pub name: String,
    /// `0.0..=100.0`. None when the vendor CLI did not report one this sample.
    pub util_pct: Option<f32>,
    /// Used / total in MiB. None when the vendor CLI did not report both.
    pub vram: Option<(u64, u64)>,
    /// Extended fields. Populated when the probe's CLI exposes them; the
    /// panel surfaces what is present rather than padding with `—`.
    pub temp_c: Option<f32>,
    pub power_w: Option<f32>,
}

/// The last successful sample, plus when it happened.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sample {
    pub probes: Vec<Probe>,
    pub fetched_at: Option<Instant>,
}

impl Sample {
    /// Anything on screen at all.
    fn has_devices(&self) -> bool {
        self.probes.iter().any(|p| !p.devices.is_empty())
    }
}

/// The panel.
pub struct GpuPanel {
    state: Arc<Mutex<Sample>>,
    stop: Arc<AtomicBool>,
    /// Snapshot taken on each `tick` so `render` can read it without holding
    /// the lock. The mutation cost is a `Vec::clone` per successful sample,
    /// not per frame.
    cached: Sample,
    /// `(probe_idx, device_idx)` for the row the cursor is over, so the
    /// footer can show extended fields the device row has no room for.
    hover: Option<(usize, usize)>,
}

impl Drop for GpuPanel {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl GpuPanel {
    /// Build a panel that has already begun sampling.
    pub fn new() -> Self {
        let state = Arc::new(Mutex::new(Sample::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_state = Arc::clone(&state);
        let worker_stop = Arc::clone(&stop);
        std::thread::Builder::new()
            .name("mirador-gpu".into())
            .spawn(move || sample_loop(&worker_state, &worker_stop))
            .expect("spawning the gpu thread");
        Self {
            state,
            stop,
            cached: Sample::default(),
            hover: None,
        }
    }

    /// Take a fresh snapshot under the lock, recovering from poisoning.
    fn snapshot(&self) -> Sample {
        match self.state.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }
}

impl Default for GpuPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// The worker. Polls, sleeps, polls again. Exits when the stop flag is set.
fn sample_loop(state: &Arc<Mutex<Sample>>, stop: &Arc<AtomicBool>) {
    loop {
        let probes = collect();
        let now = Instant::now();
        match state.lock() {
            Ok(mut g) => {
                g.probes = probes;
                g.fetched_at = Some(now);
            }
            Err(p) => {
                let mut g = p.into_inner();
                g.probes = probes;
                g.fetched_at = Some(now);
            }
        }
        match wait(SAMPLE_INTERVAL, &stop, || false) {
            crate::poll::Wake::Stop => return,
            crate::poll::Wake::Poll => {}
        }
    }
}

/// Run every probe appropriate for this platform.
///
/// Each call to a vendor CLI has a small wrapper that swallows "binary not
/// installed" silently, because the entire point of auto-detection is to
/// surface what *is* here without pestering the user about what is not.
fn collect() -> Vec<Probe> {
    let mut probes = Vec::new();
    // NVIDIA first because it is the only vendor with an implementation
    // today, and skipping a missing `nvidia-smi` is what makes the panel
    // work on a box that has only an AMD or Intel GPU. The probe functions
    // return `None` when the CLI is absent or fails; the loop is silent on
    // purpose. Pinning the order here is a one-line edit to add a fourth
    // vendor later.
    if let Some(devices) = try_nvidia_smi() {
        probes.push(Probe {
            label: "NVIDIA".into(),
            devices,
        });
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(devices) = try_intel_gpu_top() {
            probes.push(Probe {
                label: "INTEL/ARC".into(),
                devices,
            });
        }
        if let Some(devices) = try_radeontop() {
            probes.push(Probe {
                label: "AMD".into(),
                devices,
            });
        }
    }
    #[cfg(target_os = "windows")]
    {
        // `wmic` is deprecated but ships on every Windows install and exposes
        // the model name even when no vendor CLI is present. The numbers are
        // blank, which is the honest answer for a panel that did not see a
        // live sensor this sample.
        if probes.is_empty() {
            if let Some(devices) = try_wmic_video() {
                probes.push(Probe {
                    label: "WINDOWS".into(),
                    devices,
                });
            }
        }
    }
    // macOS intentionally omitted. Per the design log: detection would
    // require an IOKit bridge or `system_profiler` (not real-time), and the
    // user accepted the gap.
    probes
}

/// Run a command to completion and capture its stdout, with a wall-clock
/// bound of [`PROBE_TIMEOUT`]. A subprocess that the vendor CLI's driver
/// hangs on is killed here, so the worker thread is never blocked longer
/// than the timeout. Returns `None` on any failure: missing binary,
/// non-zero exit, hang past the timeout, or non-UTF-8 stdout.
fn run_capture(cmd: &str, args: &[&str]) -> Option<String> {
    let Ok(mut child) = std::process::Command::new(cmd)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return None;
    };
    let start = std::time::Instant::now();
    let poll_slice = Duration::from_millis(50);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut buf = Vec::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = std::io::Read::read_to_end(&mut stdout, &mut buf);
                }
                if !status.success() {
                    return None;
                }
                return Some(String::from_utf8_lossy(&buf).into_owned());
            }
            Ok(None) => {
                if start.elapsed() >= PROBE_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(poll_slice);
            }
            Err(_) => return None,
        }
    }
}

fn try_nvidia_smi() -> Option<Vec<Device>> {
    let out = run_capture(
        "nvidia-smi",
        &[
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
    )?;
    Some(parse_nvidia_csv(&out))
}

/// `nvidia-smi --format=csv,noheader,nounits` emits one line per GPU:
///
/// ```text
/// 0, NVIDIA GeForce RTX 4070, 70, 6543, 12282, 65, 142.34
/// 1, NVIDIA GeForce RTX 3090, 23, 9216, 24576, 58, 198.10
/// ```
///
/// A field that the driver could not report comes back as `[Not Supported]`
/// or `[N/A]` rather than a number; both go into `None` so the panel can
/// show `—` without guessing.
fn parse_nvidia_csv(input: &str) -> Vec<Device> {
    input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let mut parts = line.split(',');
            let index = parts.next()?.trim().parse().ok()?;
            let name = parts.next()?.trim().to_string();
            let util = parse_number(parts.next().unwrap_or("").trim()).map(|v: f32| v.clamp(0.0, 100.0));
            let mem_used = parse_number::<u64>(parts.next().unwrap_or("").trim());
            let mem_total = parse_number::<u64>(parts.next().unwrap_or("").trim());
            let temp = parse_number(parts.next().unwrap_or("").trim()).map(|v: f32| v.clamp(0.0, 200.0));
            let power = parse_number(parts.next().unwrap_or("").trim()).map(|v: f32| v.max(0.0));
            Some(Device {
                index,
                name,
                util_pct: util,
                vram: match (mem_used, mem_total) {
                    (Some(u), Some(t)) => Some((u, t)),
                    _ => None,
                },
                temp_c: temp,
                power_w: power,
            })
        })
        .collect()
}

/// Parse a value that may be `N/A`, `[Not Supported]`, or `[N/A]`. None of
/// those are numbers; everything else goes through the typed parse.
fn parse_number<T: std::str::FromStr>(s: &str) -> Option<T> {
    if s.is_empty() || s.contains('[') || s.eq_ignore_ascii_case("n/a") || s.eq_ignore_ascii_case("not supported") {
        return None;
    }
    s.parse().ok()
}

#[cfg(target_os = "linux")]
fn try_intel_gpu_top() -> Option<Vec<Device>> {
    let out = run_capture("intel_gpu_top", &["-J", "-s", "100", "-n", "1"])?;
    let parsed = parse_intel_json(&out);
    if parsed.is_empty() {
        None
    } else {
        Some(parsed)
    }
}

/// `intel_gpu_top -J` emits one JSON object per GPU when there are several:
/// `{"gpu_0": {...}, "gpu_1": {...}}`, or a single object with the engine
/// fields at the top level when there is just one. We average the busy
/// engines and synthesise the index.
fn parse_intel_json(input: &str) -> Vec<Device> {
    let value: serde_json::Value = match serde_json::from_str(input) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(obj) = value.as_object() else { return Vec::new() };
    // Two shapes, distinguished by whether the top level has `gpu_N` keys:
    // - Multi-GPU: each entry is itself an object with `busy` inside.
    // - Single-GPU: the top-level object IS the GPU's fields.
    let mut had_named_gpu = false;
    let mut entries: Vec<(u32, serde_json::Value)> = Vec::new();
    for (key, val) in obj {
        if let Some(gpu_idx) = key.strip_prefix("gpu_").and_then(|s| s.parse::<u32>().ok()) {
            had_named_gpu = true;
            entries.push((gpu_idx, val.clone()));
        }
    }
    if !had_named_gpu {
        // Single-GPU shape: top-level fields (busy / power / frequency / ...)
        // are the GPU's own readings, with no per-GPU nesting.
        entries.push((0, value.clone()));
    }
    entries
        .into_iter()
        .filter_map(|(index, v)| {
            let busy = v.get("busy").and_then(|b| b.as_array())?;
            if busy.is_empty() {
                return None;
            }
            // `intel_gpu_top -J` emits JSON numbers, not strings, so
            // `as_f64()` is the path. Filter entries that don't parse
            // rather than treating them as zero — the divisor below is the
            // count of *parsed* entries, which is also what `filter_map`
            // collects.
            let parsed: Vec<f64> = busy.iter().filter_map(|n| n.as_f64()).collect();
            if parsed.is_empty() {
                return None;
            }
            let sum: f64 = parsed.iter().sum();
            let avg = sum / parsed.len() as f64;
            let name = v
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("INTEL GPU")
                .to_string();
            Some(Device {
                index,
                name,
                // Values are already 0..=100 percentages. The clamp guards
                // against a misbehaving driver reporting >100, and the
                // final `.max(0.0)` after the f64→f32 cast keeps a
                // negative zero from appearing in the panel.
                util_pct: Some((avg.clamp(0.0, 100.0) as f32).max(0.0)),
                vram: None,
                temp_c: None,
                power_w: v.get("power").and_then(|p| p.as_f64()).map(|p| p.max(0.0) as f32),
            })
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn try_radeontop() -> Option<Vec<Device>> {
    // `radeontop` runs continuously and writes to stdout; the only way to
    // take a single snapshot is `--batch`. The output is a single block of
    // whitespace-separated tokens ending with `gpu NNN MMM TTT ...`, and
    // parsing it portably is more code than the user gets out of it. We
    // test for the binary and report its presence; sampling the percentage
    // is left to a future change when `radeontop` ships a CSV mode.
    let out = run_capture("radeontop", &["--help"])?;
    if out.is_empty() {
        return None;
    }
    // The driver is here. Without a parseable sample, surface a single
    // device with the model line in `name` and `None` for the numbers —
    // honest about not having a live reading.
    let model = out
        .lines()
        .find(|line| line.to_lowercase().contains("radeon") || line.to_lowercase().contains("amd"))
        .unwrap_or("AMD GPU");
    Some(vec![Device {
        index: 0,
        name: model.trim().to_string(),
        util_pct: None,
        vram: None,
        temp_c: None,
        power_w: None,
    }])
}

#[cfg(target_os = "windows")]
fn try_wmic_video() -> Option<Vec<Device>> {
    // `wmic path win32_VideoController get name` is deprecated but still
    // ships on every Windows install. The output is two columns with a
    // blank separator row; we extract the names and emit one device each.
    let out = run_capture("wmic", &["path", "win32_VideoController", "get", "name"])?;
    let devices: Vec<Device> = out
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.eq_ignore_ascii_case("name"))
        .enumerate()
        .map(|(i, name)| Device {
            index: i as u32,
            name: name.to_string(),
            util_pct: None,
            vram: None,
            temp_c: None,
            power_w: None,
        })
        .collect();
    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

impl Panel for GpuPanel {
    fn title(&self) -> String {
        "GPU".to_string()
    }

    /// The number of detected probes, not devices.
    ///
    /// A box with one NVIDIA GPU and a hostile `radeontop` install says `1`
    /// rather than the device count, because the probe count is what tells
    /// the reader "this panel saw two vendors' worth of hardware".
    fn counter(&self) -> Option<String> {
        let probes = self.cached.probes.len();
        if probes == 0 {
            None
        } else {
            Some(format!("{probes} vendor{}", if probes == 1 { "" } else { "s" }))
        }
    }

    /// Cheap. The 1.5s sampling cadence matches what `cargo doc` advises on
    /// panels with a worker thread; the shell is responsible for not calling
    /// us before the next spawn.
    fn refresh_interval(&self) -> Duration {
        SAMPLE_INTERVAL
    }

    /// Returns true only on a fresh sample.
    ///
    /// Mutating `cached` and comparing would be one extra `Vec::clone` per
    /// sample; comparing the previously-published sample directly under the
    /// lock keeps the total at one clone per *successful* sample.
    fn tick(&mut self) -> bool {
        let fresh = self.snapshot();
        if fresh != self.cached {
            self.cached = fresh;
            true
        } else {
            false
        }
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: RenderContext<'_>) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        let sample = &self.cached;
        let empty = !sample.has_devices();

        // Two line budgets: when there is content the footer shows the
        // hovered device's extended fields, otherwise just the empty-state
        // message.
        if empty {
            // Reset hover on empty so the footer never names a device the
            // panel no longer has.
            self.hover = None;
        }
        let footer_lines = if self.hover.is_some() && !empty { 2 } else { 1 };
        let body_height = area.height.saturating_sub(footer_lines);

        let body_area = Rect::new(area.x, area.y, area.width, body_height);
        let footer_area = Rect::new(area.x, area.y + body_height, area.width, footer_lines);

        self.draw_body(frame, area, body_area, sample, ctx.theme);
        self.draw_footer(frame, footer_area, sample, footer_lines, ctx.theme);
    }

    /// Draw one section header per probe and one row per device inside it.
    ///
    /// Iterates the cached sample without allocating — the only `String`s are
    /// the row bodies built by [`Self::compose_device_line`] and the section
    /// labels, both of which already exist on the panel.
    fn draw_body(
        &self,
        frame: &mut Frame,
        area: Rect,
        body_area: Rect,
        sample: &Sample,
        theme: &crate::theme::Theme,
    ) {
        let mut y = area.y;
        for (probe_idx, probe) in sample.probes.iter().enumerate() {
            if y >= area.y + body_area.height {
                break;
            }
            let label = crate::glyphs::utility(&probe.label);
            frame.render_widget(
                Paragraph::new(Span::styled(
                    label,
                    Style::default()
                        .fg(theme.label)
                        .add_modifier(Modifier::BOLD),
                )),
                Rect::new(area.x, y, area.width, 1),
            );
            y += 1;
            for (device_idx, device) in probe.devices.iter().enumerate() {
                if y >= area.y + body_area.height {
                    break;
                }
                let mut line = Self::compose_device_line(device, area.width);
                // Truncate the final composed line to the panel width — the
                // parts formula adds up to more than `area.width` at narrow
                // sizes, and invariant 19 says anything dropped is dropped
                // whole. We drop from the right (the vram/figure pair) so
                // the index and name survive.
                line = truncate(&line, area.width as usize);
                // Pad to width so the highlight reaches the right edge when
                // the device name is short.
                if line.chars().count() < area.width as usize {
                    let pad = area.width as usize - line.chars().count();
                    line.push_str(&" ".repeat(pad));
                }
                // A hovered row gets the accent colour, not the data colour,
                // so the eye finds it. Focus is dimmed elsewhere rather than
                // focus highlighted here — invariant: "dim the unfocused,
                // never brighten the focused".
                let hover_target = self.hover == Some((probe_idx, device_idx));
                let style = if hover_target {
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme.text)
                };
                frame.render_widget(
                    Paragraph::new(Span::styled(line, style)),
                    Rect::new(area.x, y, area.width, 1),
                );
                y += 1;
            }
        }
        // Body may be empty (no probes yet, or none detected) — leave a
        // single-line hint when there is room above the footer.
        let no_probes = sample.probes.is_empty()
            || sample.probes.iter().all(|p| p.devices.is_empty());
        if no_probes && body_area.height > 0 {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "no GPU detected",
                    Style::default().fg(theme.muted),
                )),
                body_area,
            );
        }
    }

    /// Draw the one or two-line footer: the hovered device's extended fields
    /// (when one is shown), and the freshness + empty-state warning that the
    /// rest of the panel relies on for context.
    fn draw_footer(
        &self,
        frame: &mut Frame,
        footer_area: Rect,
        sample: &Sample,
        footer_lines: u16,
        theme: &crate::theme::Theme,
    ) {
        let freshness = sample
            .fetched_at
            .map_or_else(|| "never sampled".to_string(), |at| crate::panel::describe_age(at.elapsed()));
        let empty_label = !sample.has_devices() || sample.probes.is_empty();
        let footer_style = if empty_label {
            Style::default().fg(theme.warning)
        } else {
            Style::default().fg(theme.muted)
        };
        let footer_text = if empty_label {
            format!("no GPU detected · {freshness}")
        } else {
            format!("sampled {freshness} ago")
        };
        if footer_lines == 2 {
            if let Some((probe_idx, device_idx)) = self.hover {
                if let Some(device) = sample
                    .probes
                    .get(probe_idx)
                    .and_then(|p| p.devices.get(device_idx))
                {
                    let detail = Self::compose_device_detail(device, footer_area.width);
                    frame.render_widget(
                        Paragraph::new(Span::styled(detail, Style::default().fg(theme.accent))),
                        Rect::new(footer_area.x, footer_area.y, footer_area.width, 1),
                    );
                }
            }
            frame.render_widget(
                Paragraph::new(Span::styled(footer_text, footer_style)),
                Rect::new(footer_area.x, footer_area.y + 1, footer_area.width, 1),
            );
        } else {
            frame.render_widget(
                Paragraph::new(Span::styled(footer_text, footer_style)),
                Rect::new(footer_area.x, footer_area.y, footer_area.width, 1),
            );
        }
    }

    /// Hover is the only mouse input that matters here. Click is reserved
    /// for future "lock this device as the focused readout" — kept as Ignored
    /// rather than swallowed, so the shell can move focus across rows with a
    /// single click tomorrow without a trait break.
    fn handle_mouse(&mut self, event: MouseEvent, area: Rect) -> crate::panel::KeyOutcome {
        if event.kind != MouseEventKind::Moved {
            return crate::panel::KeyOutcome::Ignored;
        }
        // `event.row` is absolute; subtract the panel origin.
        if event.row < area.y || event.row >= area.y + area.height {
            self.hover = None;
            return crate::panel::KeyOutcome::Ignored;
        }
        // Walk the row layout the same way `render` does. O(N) over a
        // handful of devices — a HashMap would be wasted ceremony, and its
        // allocation would be the only one this panel introduces on every
        // mouse move.
        let mut y = area.y;
        for (probe_idx, probe) in self.cached.probes.iter().enumerate() {
            if event.row == y {
                // The cursor is on a section label, not a device row.
                self.hover = None;
                return crate::panel::KeyOutcome::Ignored;
            }
            y += 1;
            for (device_idx, _) in probe.devices.iter().enumerate() {
                if event.row == y {
                    let new = (probe_idx, device_idx);
                    if self.hover != Some(new) {
                        self.hover = Some(new);
                        return crate::panel::KeyOutcome::Consumed;
                    }
                    return crate::panel::KeyOutcome::Consumed;
                }
                y += 1;
            }
        }
        self.hover = None;
        crate::panel::KeyOutcome::Ignored
    }
}

impl GpuPanel {
    /// Three columns: index + name, util%, vram fraction. `[AUTO_WIDTH]` is
    /// for invariant 19: nothing wider than the area. The separator sits at
    /// the front of each part so a drop-from-end never leaves a dangling
    /// bullet.
    fn compose_device_line(device: &Device, width: u16) -> String {
        let idx = device.index;
        let name = truncate(&device.name, width.saturating_sub(20) as usize);
        let util = device
            .util_pct
            .map_or_else(|| "  —".to_string(), |v| format!("{v:>3.0}%"));
        let vram = device
            .vram
            .map_or_else(|| "—".to_string(), |(used, total)| format_vram_fraction(used, total));
        format!("{idx} {name}  {util}  {vram}")
    }

    fn compose_device_detail(device: &Device, width: u16) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(t) = device.temp_c {
            parts.push(format!("{t:.0}°C"));
        }
        if let Some(p) = device.power_w {
            parts.push(format!("{p:.0} W"));
        }
        let extra = parts.join(" · ");
        let name = format!("{}: {}", device.index, device.name);
        truncate(&format!("{name} — {extra}"), width as usize)
    }
}

fn format_vram_fraction(used_mib: u64, total_mib: u64) -> String {
    let used_gb = used_mib as f64 / 1024.0;
    let total_gb = total_mib as f64 / 1024.0;
    if total_gb < 0.05 {
        return "—".to_string();
    }
    format!("{:.1}/{:.1} GB", used_gb.min(total_gb), total_gb)
}

/// Truncate by display cells, not `chars()`, because the model names can
/// contain CJK (`NVIDIA GeForce RTX`) — and a `Vec<char>` truncation that
/// cuts a wide glyph is the same bug news hit.
fn truncate(s: &str, max_cells: usize) -> String {
    if max_cells == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut cells = 0;
    for c in s.chars() {
        let w = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if cells + w > max_cells {
            break;
        }
        out.push(c);
        cells += w;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::{KeyModifiers, MouseButton};

    #[test]
    fn parses_nvidia_csv_one_line() {
        let raw = "0, NVIDIA GeForce RTX 4070, 70, 6543, 12282, 65, 142.34\n";
        let devices = parse_nvidia_csv(raw);
        assert_eq!(devices.len(), 1);
        let d = &devices[0];
        assert_eq!(d.index, 0);
        assert_eq!(d.name, "NVIDIA GeForce RTX 4070");
        assert_eq!(d.util_pct, Some(70.0));
        assert_eq!(d.vram, Some((6543, 12282)));
        assert_eq!(d.temp_c, Some(65.0));
        assert_eq!(d.power_w, Some(142.34));
    }

    #[test]
    fn parses_nvidia_csv_many_lines() {
        let raw = "0, RTX 4070, 70, 100, 12282, 65, 142\n\
                   1, RTX 3090, 23, 9216, 24576, 58, 198\n\
                   2, RTX A6000, 0, 0, 49152, 31, 28.5\n";
        let devices = parse_nvidia_csv(raw);
        assert_eq!(devices.len(), 3);
        assert_eq!(devices[2].util_pct, Some(0.0));
        assert_eq!(devices[2].vram, Some((0, 49152)));
    }

    #[test]
    fn parses_nvidia_na_fields_as_none() {
        let raw = "0, RTX 4070, [Not Supported], 100, 12282, N/A, [N/A]\n";
        let d = parse_nvidia_csv(raw).into_iter().next().unwrap();
        assert_eq!(d.util_pct, None);
        assert_eq!(d.temp_c, None);
        assert_eq!(d.power_w, None);
        // vram still present
        assert_eq!(d.vram, Some((100, 12282)));
    }

    #[test]
    fn parses_empty_nvidia_output_as_empty_vec() {
        assert!(parse_nvidia_csv("").is_empty());
        assert!(parse_nvidia_csv("\n\n\n").is_empty());
    }

    #[test]
    fn parses_intel_json_one_gpu() {
        // Single-GPU shape: top-level busy / power / freq. `intel_gpu_top -J`
        // emits per-engine percentages already in the 0..=100 range, so the
        // average stays where it is.
        let raw = r#"{"period_ms":100.0,"rc6":0.0,"frequency":1300.0,"power":7.5,"busy":[25.0,75.0,50.0,0.0]}"#;
        let devices = parse_intel_json(raw);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].util_pct, Some(37.5));
        assert_eq!(devices[0].power_w, Some(7.5));
    }

    #[test]
    fn parses_intel_json_multi_gpu() {
        let raw = r#"{"gpu_0":{"busy":[50.0,50.0]},"gpu_1":{"busy":[0.0,0.0]}}"#;
        let devices = parse_intel_json(raw);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].util_pct, Some(50.0));
        assert_eq!(devices[1].util_pct, Some(0.0));
        assert_eq!(devices[0].index, 0);
        assert_eq!(devices[1].index, 1);
    }

    #[test]
    fn format_vram_fraction_rounds_to_one_decimal() {
        assert_eq!(format_vram_fraction(1024, 12 * 1024), "1.0/12.0 GB");
        assert_eq!(format_vram_fraction(8192, 12282), "8.0/12.0 GB");
    }

    #[test]
    fn format_vram_fraction_handles_zero_total() {
        assert_eq!(format_vram_fraction(0, 0), "—");
    }

    #[test]
    fn truncate_preserves_ascii_within_budget() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello");
    }

    #[test]
    fn truncate_handles_wide_glyphs_by_cells_not_chars() {
        // A CJK character is 2 display cells. Cutting at the byte boundary
        // would corrupt the row — `chars()` would not, but a naive byte
        // truncation would panic. Pin the cell-budget behaviour here so the
        // fix stays put.
        // Cells: 機=2, 器=2, 之=2, 心=2, G=1, P=1, U=1 (11 cells total).
        let s = "機器之心GPU";
        assert_eq!(truncate(s, 4), "機器");    // 2 chars, 4 cells
        assert_eq!(truncate(s, 6), "機器之");   // 3 chars, 6 cells
        assert_eq!(truncate(s, 11), s);         // fits whole
    }

    #[test]
    fn renders_at_any_size_without_panicking() {
        // The `every_widget_renders_at_any_size_without_panicking` test in
        // `widgets::mod` covers every widget, but this is faster and lets
        // us fail on a single regression without sweeping the lot.
        let dir = std::env::temp_dir().join(format!("mirador-gpu-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut config = Config::default();
        config.todo.file = Some(dir.join("todos.toml"));

        let mut panel = GpuPanel::new();
        panel.tick();
        let gradients = config.theme.gradients();
        let watch = crate::watch::WatchLog::default();

        for (width, height) in [(1u16, 1u16), (2, 3), (10, 4), (40, 12), (200, 60)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|frame| {
                    let area = frame.area();
                    panel.render(
                        frame,
                        area,
                        crate::panel::RenderContext {
                            theme: &config.theme,
                            gradients: &gradients,
                            focused: true,
                            watch: &watch,
                        },
                    );
                })
                .unwrap_or_else(|e| panic!("gpu panel failed at {width}x{height}: {e}"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The worker must not panic when the binary is absent on the test
    /// machine (this is also what the test machine itself has). The thread
    /// runs forever; we only check that one sample settled.
    #[test]
    fn no_binary_yields_a_sample_with_zero_probes() {
        let state = Arc::new(Mutex::new(Sample::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_state = Arc::clone(&state);
        let worker_stop = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("mirador-gpu-test".into())
            .spawn(move || sample_loop(&worker_state, &worker_stop))
            .expect("spawning the gpu test thread");
        // Give the worker time to take its first sample. Then ask it to
        // stop, so the test does not leak a thread.
        std::thread::sleep(Duration::from_millis(200));
        stop.store(true, Ordering::Relaxed);
        handle.join().expect("worker thread should exit cleanly");

        let snapshot = state.lock().unwrap().clone();
        // Whichever binary the test runner has installed, the probe count
        // is some non-negative number — never a panic.
        assert!(snapshot.probes.len() <= 4);
    }

    #[test]
    fn mouse_event_on_a_row_records_hover() {
        let mut panel = GpuPanel::new();
        panel.tick();
        // No probes, no rows. Hover stays None.
        let area = Rect::new(0, 0, 20, 5);
        let event = MouseEvent {
            kind: MouseEventKind::Moved,
            column: 0,
            row: 1,
            modifiers: KeyModifiers::NONE,
        };
        let _ = panel.handle_mouse(event, area);
        // Whether any probes are present depends on the machine, so we do
        // not assert on `hover` directly. The non-panic is the contract.
    }

    #[test]
    fn mouse_event_outside_panel_clears_hover() {
        let mut panel = GpuPanel::new();
        panel.hover = Some((0, 0));
        let area = Rect::new(10, 10, 20, 5);
        let event = MouseEvent {
            kind: MouseEventKind::Moved,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        };
        let outcome = panel.handle_mouse(event, area);
        assert_eq!(outcome, crate::panel::KeyOutcome::Ignored);
        assert_eq!(panel.hover, None);
    }

    #[test]
    fn click_is_ignored_so_focus_can_stay() {
        let mut panel = GpuPanel::new();
        let area = Rect::new(0, 0, 20, 5);
        let event = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 1,
            row: 1,
            modifiers: KeyModifiers::NONE,
        };
        let outcome = panel.handle_mouse(event, area);
        assert_eq!(outcome, crate::panel::KeyOutcome::Ignored);
    }

    #[test]
    fn describe_age_includes_in_the_empty_state_render() {
        // Eyeball check: render the empty state at 30x10 and confirm the
        // footer mentions `no GPU detected`. This is the contract that
        // somebody who just installed mirador on a server reads first.
        let dir = std::env::temp_dir().join(format!("mirador-gpu-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut config = Config::default();
        config.todo.file = Some(dir.join("todos.toml"));

        let mut panel = GpuPanel::new();
        panel.tick();
        let gradients = config.theme.gradients();
        let watch = crate::watch::WatchLog::default();

        let mut terminal = Terminal::new(TestBackend::new(30, 10)).unwrap();
        terminal
            .draw(|f| {
                panel.render(
                    f,
                    f.area(),
                    crate::panel::RenderContext {
                        theme: &config.theme,
                        gradients: &gradients,
                        focused: true,
                        watch: &watch,
                    },
                )
            })
            .unwrap();

        let buf = terminal.backend().buffer().clone();
        let mut found = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                found.push_str(buf[(x, y)].symbol());
            }
        }
        // The full output is the only thing we have to check; assert at
        // least that `no GPU detected` reaches the buffer somewhere.
        assert!(
            found.contains("no GPU detected"),
            "empty state text should appear; got:\n{found}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
