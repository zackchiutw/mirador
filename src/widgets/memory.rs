//! Memory utilisation: used RAM as a percentage, with a scrolling history chart.
//!
//! Uses the same gradient as the CPU panel (`cpu_gradient`), so the instrument
//! row changes temperature together. Swap is shown when the panel has height and
//! swap exists on the system.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::Paragraph;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

use crate::chart::BrailleGraph;
use crate::config::MemoryConfig;
use crate::panel::{Panel, RenderContext};
use crate::samples::push_bounded;

/// The memory panel.
pub struct MemoryPanel {
    config: MemoryConfig,
    system: System,
    /// Recent used-percentage samples, oldest first.
    history: VecDeque<u64>,
    /// Used RAM in bytes from the most recent sample.
    used: u64,
    /// Total RAM in bytes.
    total: u64,
    /// Used swap in bytes, zero when there is no swap.
    swap_used: u64,
    /// Total swap in bytes, zero when there is no swap.
    swap_total: u64,
    /// `None` until the first sample, so it fires immediately.
    last_sample: Option<Instant>,
    /// Cells the graph was last drawn into, so the history can grow to fill it.
    graph_cells: usize,
}

impl std::fmt::Debug for MemoryPanel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPanel")
            .field("used", &self.used)
            .field("total", &self.total)
            .field("samples", &self.history.len())
            .finish_non_exhaustive()
    }
}

impl MemoryPanel {
    /// Build the panel and take a first sample.
    pub fn new(config: MemoryConfig) -> Self {
        let mut system = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
        );
        system.refresh_memory();

        let total = system.total_memory();
        let used = system.used_memory();
        let swap_total = system.total_swap();
        let swap_used = system.used_swap();

        Self {
            history: VecDeque::with_capacity(config.history.max(1)),
            config,
            system,
            used,
            total,
            swap_used,
            swap_total,
            last_sample: None,
            graph_cells: 0,
        }
    }

    /// The current used percentage, clamped to 0..=100.
    fn used_pct(&self) -> u8 {
        if self.total == 0 {
            return 0;
        }
        ((u128::from(self.used) * 100 / u128::from(self.total)).min(100)) as u8
    }

    /// Take a sample if enough time has passed since the last one.
    ///
    /// Returns whether it actually sampled; see `CpuPanel::sample` for why the
    /// answer matters more than it looks.
    fn sample(&mut self) -> bool {
        let interval = Duration::from_secs(self.config.sample_secs.max(1));

        if self.last_sample.is_some_and(|at| at.elapsed() < interval) {
            return false;
        }

        self.system
            .refresh_memory_specifics(MemoryRefreshKind::everything());
        self.used = self.system.used_memory();
        self.total = self.system.total_memory();
        self.swap_used = self.system.used_swap();
        self.swap_total = self.system.total_swap();
        self.last_sample = Some(Instant::now());

        let pct = self.used_pct();
        let capacity = self.capacity();
        push_bounded(&mut self.history, u64::from(pct), capacity);
        true
    }

    /// How many samples to retain; see [`crate::samples::capacity`].
    fn capacity(&self) -> usize {
        crate::samples::capacity(self.config.history, self.graph_cells)
    }

    /// Whether swap is worth showing: it exists and the panel has room.
    fn has_swap(&self) -> bool {
        self.swap_total > 0
    }
}

impl Panel for MemoryPanel {
    fn title(&self) -> String {
        "MEMORY".to_string()
    }

    fn counter(&self) -> Option<String> {
        let avail = self.total.saturating_sub(self.used);
        Some(format_bytes(avail))
    }

    fn refresh_interval(&self) -> Duration {
        Duration::from_millis(500)
    }

    fn tick(&mut self) -> bool {
        self.sample()
    }

    fn render(&mut self, frame: &mut Frame, area: Rect, ctx: RenderContext<'_>) {
        let theme = ctx.theme;
        if area.height == 0 || area.width == 0 {
            return;
        }

        let gradient = &ctx.gradients.cpu;
        let track = Style::default().fg(theme.track);
        let pct = self.used_pct();
        let show_swap = self.has_swap() && area.height >= 5;

        let rows = Layout::vertical([
            Constraint::Length(1),                        // readout
            Constraint::Min(1),                           // graph
            Constraint::Length(u16::from(show_swap) * 2), // swap meter
        ])
        .split(area);

        // The number takes its colour from the same ramp as the graph, so the
        // whole panel changes temperature together — same trade as the CPU.
        // `pct` is clamped to 0..=100, so the cast cannot wrap.
        let colour = gradient.at(i64::from(pct));

        // Three parts, same dropping discipline as the CPU readout: the figure
        // and its `%` are inseparable; the label and the total are droppable.
        let total_str = format_bytes(self.total);
        frame.render_widget(
            Paragraph::new(crate::grid::assemble(
                vec![
                    vec![
                        Span::styled(
                            if rows[0].width >= 6 {
                                format!("{:>5.1}", f32::from(pct))
                            } else {
                                format!("{:.1}", f32::from(pct))
                            },
                            Style::default().fg(colour).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("%", Style::default().fg(theme.muted)),
                    ],
                    vec![Span::styled(
                        format!(" {}", crate::glyphs::utility("used")),
                        Style::default()
                            .fg(theme.label)
                            .add_modifier(Modifier::BOLD),
                    )],
                    vec![Span::styled(
                        format!("   {total_str}"),
                        Style::default().fg(theme.muted),
                    )],
                ],
                rows[0].width,
            )),
            rows[0],
        );

        self.graph_cells = rows[1].width as usize;

        if rows[1].height > 0 {
            let data: Vec<u64> = self.history.iter().copied().collect();
            BrailleGraph::new(&data, 100, gradient)
                .track_style(track)
                .render(rows[1], frame.buffer_mut());
        }

        if show_swap && rows[2].height >= 2 {
            let swap_pct: u8 = if self.swap_total > 0 {
                ((u128::from(self.swap_used) * 100 / u128::from(self.swap_total)).min(100)) as u8
            } else {
                0
            };

            // Label on the first row, meter bar on the second — same shape as
            // the CPU per-core meters, but one bar for swap.
            let swap_label = format!(
                "{}  {:.1}%  {}",
                crate::glyphs::utility("swap"),
                f32::from(swap_pct),
                format_bytes(self.swap_used),
            );
            frame.render_widget(
                Paragraph::new(crate::grid::assemble(
                    vec![vec![Span::styled(
                        swap_label,
                        Style::default()
                            .fg(theme.label)
                            .add_modifier(Modifier::BOLD),
                    )]],
                    rows[2].width,
                )),
                Rect::new(rows[2].x, rows[2].y, rows[2].width, 1),
            );

            // Meter bar fills the second row, coloured by the same gradient.
            let meter_cells =
                crate::chart::meter_spans(u64::from(swap_pct), 100, rows[2].width, gradient, track);
            let y = rows[2].y + 1;
            for (index, (glyph, style)) in meter_cells.iter().enumerate() {
                let cx = rows[2].x + u16::try_from(index).unwrap_or(0);
                if cx >= rows[2].x + rows[2].width {
                    break;
                }
                frame.buffer_mut()[(cx, y)]
                    .set_char(*glyph)
                    .set_style(*style);
            }
        }
    }
}

/// Format a byte count with a binary-ish scale and one decimal place.
///
/// Shared with the network panel; duplicated here rather than adding a cross-
/// widget dependency for one function. Kept in sync by the same test suite.
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[0])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn used_pct_is_zero_when_total_is_zero() {
        let panel = MemoryPanel::new(MemoryConfig::default());
        // On a real system total > 0, but the guard must hold regardless.
        let pct = if panel.total == 0 {
            0u8
        } else {
            ((u128::from(panel.used) * 100 / u128::from(panel.total)).min(100)) as u8
        };
        assert!(pct <= 100);
    }

    #[test]
    fn history_is_bounded_by_the_configured_capacity() {
        let config = MemoryConfig {
            history: 5,
            ..Default::default()
        };
        let mut panel = MemoryPanel::new(config);
        for i in 0..50 {
            if panel.history.len() >= 5 {
                panel.history.pop_front();
            }
            panel.history.push_back(i);
        }
        assert_eq!(panel.history.len(), 5);
        assert_eq!(panel.history.front(), Some(&45));
    }

    #[test]
    fn bytes_format_across_every_unit() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(1024u64.pow(4)), "1.0 TB");
    }

    #[test]
    fn very_large_values_stay_in_the_top_unit_instead_of_overflowing() {
        let huge = format_bytes(u64::MAX);
        assert!(huge.ends_with("TB"), "got: {huge}");
    }
}
