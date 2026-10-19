#!/usr/bin/env python3
"""
Traffic-Highlighted NSFNET SDN Topology Generator
Generates a publication-grade 2D topology diagram with traffic-carrying edges
highlighted in color, based on the generateTraffic() background traffic flows
in ai-based-routing-algorithm-experiment.py.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Set MPLCONFIGDIR to a writable temporary directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'


# ---------------------------------------------------------------------------
# Icon Drawing Functions (matching generate_topology.py style)
# ---------------------------------------------------------------------------

def draw_switch_icon(ax, x, y, scale=1.0, label=""):
    """Draws a 1U Rack-mount Network Switch vector icon."""
    w = 0.65 * scale
    h = 0.22 * scale

    chassis = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor='#141414', edgecolor='#000000', linewidth=1.2, zorder=5
    )
    ax.add_patch(chassis)

    ear_w = 0.04 * scale
    ear_h = 0.16 * scale
    ax.add_patch(patches.Rectangle((x - w/2 - ear_w, y - ear_h/2), ear_w, ear_h,
                                   facecolor='#333333', edgecolor='#000000', linewidth=0.8, zorder=4))
    ax.add_patch(patches.Rectangle((x + w/2, y - ear_h/2), ear_w, ear_h,
                                   facecolor='#333333', edgecolor='#000000', linewidth=0.8, zorder=4))

    num_ports = 12
    port_w = (w * 0.68) / num_ports
    port_h = h * 0.38
    start_x = x - (w * 0.34) + port_w / 2
    port_y = y - port_h / 2

    for i in range(num_ports):
        px = start_x + i * port_w
        ax.add_patch(patches.Rectangle((px - port_w*0.4, port_y), port_w*0.78, port_h,
                                       facecolor='#ffffff', edgecolor='#222222', linewidth=0.4, zorder=6))

    led_x = x - w/2 + 0.06 * scale
    ax.plot(led_x, y + 0.03*scale, marker='o', markersize=2.5*scale, color='#ffffff', zorder=6)
    ax.plot(led_x, y - 0.03*scale, marker='o', markersize=2.5*scale, color='#ffffff', zorder=6)

    if label:
        ax.text(x, y + h/2 + 0.12*scale, label, fontsize=12*scale, fontweight='bold',
                color='#000000', ha='center', va='bottom', zorder=7)


def draw_host_icon(ax, x, y, scale=1.0, label="", sublabel=""):
    """Draws a Desktop PC / Host vector icon."""
    mw = 0.42 * scale
    mh = 0.30 * scale
    my = y + 0.06 * scale

    mon = patches.FancyBboxPatch(
        (x - mw/2, my - mh/2), mw, mh,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor='#1a1a1a', edgecolor='#000000', linewidth=1.2, zorder=5
    )
    ax.add_patch(mon)

    sw = mw * 0.82
    sh = mh * 0.75
    ax.add_patch(patches.Rectangle((x - sw/2, my - sh/2), sw, sh,
                                   facecolor='#ffffff', edgecolor='#444444', linewidth=0.6, zorder=6))
    ax.plot([x - sw*0.35, x + sw*0.35], [my + sh*0.12, my + sh*0.12],
            color='#222222', lw=1.0*scale, zorder=7)
    ax.plot([x - sw*0.35, x + sw*0.05], [my - sh*0.15, my - sh*0.15],
            color='#666666', lw=1.0*scale, zorder=7)

    stand_w = 0.06 * scale
    stand_h = 0.08 * scale
    stand_y = my - mh/2 - stand_h
    ax.add_patch(patches.Rectangle((x - stand_w/2, stand_y), stand_w, stand_h,
                                   facecolor='#333333', edgecolor='#000000', linewidth=0.6, zorder=4))
    base_w = 0.24 * scale
    base_h = 0.03 * scale
    ax.add_patch(patches.Rectangle((x - base_w/2, stand_y - base_h), base_w, base_h,
                                   facecolor='#1a1a1a', edgecolor='#000000', linewidth=0.6, zorder=4))

    if label:
        ax.text(x, y - 0.22*scale, label, fontsize=10.5*scale, fontweight='bold',
                color='#000000', ha='center', va='top', zorder=7)
    if sublabel:
        ax.text(x, y - 0.36*scale, sublabel, fontsize=8.5*scale, fontweight='bold',
                color='#444444', ha='center', va='top', zorder=7)


def draw_server_icon(ax, x, y, scale=1.0, label="", sublabel=""):
    """Draws a Vertical Enterprise Server Tower vector icon."""
    w = 0.32 * scale
    h = 0.52 * scale

    chassis = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor='#141414', edgecolor='#000000', linewidth=1.2, zorder=5
    )
    ax.add_patch(chassis)

    num_bays = 4
    bay_h = (h * 0.65) / num_bays
    bay_w = w * 0.78
    start_y = y + (h * 0.30) - bay_h / 2

    for i in range(num_bays):
        by = start_y - i * (bay_h * 1.1)
        ax.add_patch(patches.Rectangle((x - bay_w/2, by), bay_w, bay_h,
                                       facecolor='#3a3a3a', edgecolor='#000000', linewidth=0.6, zorder=6))
        ax.plot([x - bay_w*0.35, x + bay_w*0.1], [by + bay_h/2, by + bay_h/2],
                color='#ffffff', lw=0.8*scale, zorder=7)
        ax.plot(x + bay_w*0.3, by + bay_h/2, marker='s', markersize=2.2*scale, color='#ffffff', zorder=7)

    ax.plot(x, y - h*0.38, marker='o', markersize=4.0*scale, color='#ffffff', zorder=7)

    if label:
        ax.text(x, y - h/2 - 0.08*scale, label, fontsize=10.5*scale, fontweight='bold',
                color='#000000', ha='center', va='top', zorder=7)
    if sublabel:
        ax.text(x, y - h/2 - 0.22*scale, sublabel, fontsize=8.5*scale, fontweight='bold',
                color='#444444', ha='center', va='top', zorder=7)


def draw_controller_icon(ax, x, y, scale=1.0, label="Floodlight SDN Controller"):
    """Draws an SDN Controller box icon."""
    w = 0.60 * scale
    h = 0.38 * scale

    box = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor='#ffffff', edgecolor='#000000', linewidth=1.4, zorder=5
    )
    ax.add_patch(box)

    hdr_h = 0.10 * scale
    hdr = patches.Rectangle((x - w/2 + 0.02*scale, y + h/2 - hdr_h - 0.02*scale),
                             w - 0.04*scale, hdr_h, facecolor='#000000', zorder=6)
    ax.add_patch(hdr)
    ax.text(x, y + h/2 - hdr_h/2 - 0.02*scale, "SDN", color='#ffffff',
            fontweight='bold', fontsize=9*scale, ha='center', va='center', zorder=7)

    num_sliders = 3
    slider_x_start = x - w * 0.25
    slider_dx = w * 0.25
    for i in range(num_sliders):
        sx = slider_x_start + i * slider_dx
        sy_top = y + h * 0.06
        sy_bot = y - h * 0.36
        ax.plot([sx, sx], [sy_bot, sy_top], color='#000000', lw=1.2*scale, zorder=6)
        knob_y = y - h*0.18 + i*0.06*scale
        ax.add_patch(patches.Rectangle((sx - 0.04*scale, knob_y - 0.025*scale),
                                       0.08*scale, 0.05*scale, facecolor='#000000', zorder=7))

    if label:
        ax.text(x, y + h/2 + 0.08*scale, label, fontsize=12*scale, fontweight='bold',
                color='#000000', ha='center', va='bottom', zorder=7)


# ---------------------------------------------------------------------------
# Main Topology + Traffic Highlight Generator
# ---------------------------------------------------------------------------

def generate_traffic_topology(output_path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_facecolor('#ffffff')

    # ---- Switch Positions (matching generate_topology.py layout) ----
    sw_pos = {
        's0':  (0.8,   4.0),
        's1':  (2.8,   6.2),
        's2':  (2.6,   1.8),
        's3':  (4.5,   4.4),
        's4':  (5.8,   3.2),
        's5':  (5.6,   0.8),
        's6':  (7.6,   4.0),
        's7':  (9.2,   4.6),
        's8':  (10.5,  6.6),
        's9':  (14.2,  4.8),
        's10': (11.4,  4.4),
        's11': (12.2,  7.6),
        's12': (12.6,  2.4),
        's13': (10.6,  0.4),
    }

    # ---- All Inter-Switch backbone links ----
    all_switch_links = [
        ('s0', 's1'), ('s0', 's2'), ('s0', 's3'),
        ('s1', 's7'), ('s1', 's2'),
        ('s2', 's5'),
        ('s3', 's4'), ('s3', 's8'),
        ('s4', 's5'), ('s4', 's6'),
        ('s5', 's12'), ('s5', 's13'),
        ('s6', 's7'),
        ('s7', 's10'),
        ('s8', 's9'), ('s8', 's11'),
        ('s9', 's10'), ('s9', 's12'),
        ('s10', 's11'), ('s10', 's13'),
        ('s11', 's12')
    ]

    # ---- Host Positions ----
    # (position, attached_switch, label, sublabel, is_server)
    host_info = {
        'h0':  ((-0.5, 4.0),   's0',  'h0',   '(Receiver)', False),
        'h1':  ((2.2, 7.6),    's1',  'h1',   '',           False),
        'h12': ((3.4, 7.6),    's1',  'h12',  '',           False),
        'h2':  ((1.8, 0.7),    's2',  'h2',   '',           False),
        'h16': ((3.2, 0.7),    's2',  'h16',  '',           False),
        'h3':  ((4.0, 5.7),    's3',  'h3',   '',           False),
        'h11': ((5.2, 5.7),    's3',  'h11',  '',           False),
        'h19': ((3.8, 3.2),    's3',  'h19',  '',           False),
        'h4':  ((6.5, 2.2),    's4',  'h4',   '',           False),
        'h5':  ((4.5, -0.4),   's5',  'h5',   '',           False),
        'h18': ((6.3, -0.4),   's5',  'h18',  '',           False),
        'h6':  ((8.4, 5.8),    's7',  'h6',   '',           False),
        'h14': ((9.4, 5.8),    's7',  'h14',  '',           False),
        'h7':  ((9.5, 7.9),    's8',  'h7',   '',           False),
        'h13': ((10.7, 7.9),   's8',  'h13',  '',           False),
        'h8':  ((15.5, 4.8),   's9',  'h8',   '(Sender)',   True),
        'h9':  ((11.4, 3.2),   's10', 'h9',   '',           False),
        'h17': ((12.5, 3.2),   's10', 'h17',  '',           False),
        'h10': ((11.4, 8.8),   's11', 'h10',  '',           False),
        'h15': ((12.8, 8.8),   's11', 'h15',  '',           False),
    }

    # -----------------------------------------------------------------------
    # Traffic Flows from generateTraffic() in startNetwork()
    #
    # Traced from setTrafficPaths() flow rules:
    #   h7(s8) <-> h3(s3):   s8 <-> s3            [100M]
    #   h13(s8) <-> h11(s3): s8 <-> s3            [100M]
    #   h6(s7) <-> h1(s1):   s7 <-> s1            [100M]
    #   h14(s7) <-> h12(s1): s7 <-> s1            [100M]
    #   h9(s10) <-> h10(s11): s10 <-> s11         [100M]
    #   h17(s10) <-> h15(s11): s10 <-> s11        [100M]
    #   h5(s5) <-> h2(s2):   s5 <-> s2            [100M]
    #   h18(s5) <-> h16(s2): s5 <-> s2            [100M]
    #   h4(s4) <-> h19(s3):  s4 <-> s3            [3M]
    # -----------------------------------------------------------------------

    COLORS = {
        'high_traffic':  '#E53935',   # Red — heavy 100M flows
        'low_traffic':   '#FB8C00',   # Orange — light 3M flow
        'idle':          '#BDBDBD',   # Light gray — no traffic
    }

    traffic_on_switch_link = {}
    traffic_on_host_link = {}

    def add_sw_traffic(sA, sB, color, bw_label):
        key = frozenset([sA, sB])
        traffic_on_switch_link.setdefault(key, []).append((color, bw_label))

    def add_host_traffic(host, sw, color, bw_label):
        key = (host, sw)
        traffic_on_host_link.setdefault(key, []).append((color, bw_label))

    # --- 100M Traffic Pairs (Red) ---
    # h7(s8) <-> h3(s3)
    add_sw_traffic('s8', 's3', COLORS['high_traffic'], '100M')
    add_host_traffic('h7', 's8', COLORS['high_traffic'], '100M')
    add_host_traffic('h3', 's3', COLORS['high_traffic'], '100M')

    # h13(s8) <-> h11(s3)
    add_sw_traffic('s8', 's3', COLORS['high_traffic'], '100M')
    add_host_traffic('h13', 's8', COLORS['high_traffic'], '100M')
    add_host_traffic('h11', 's3', COLORS['high_traffic'], '100M')

    # h6(s7) <-> h1(s1)
    add_sw_traffic('s7', 's1', COLORS['high_traffic'], '100M')
    add_host_traffic('h6', 's7', COLORS['high_traffic'], '100M')
    add_host_traffic('h1', 's1', COLORS['high_traffic'], '100M')

    # h14(s7) <-> h12(s1)
    add_sw_traffic('s7', 's1', COLORS['high_traffic'], '100M')
    add_host_traffic('h14', 's7', COLORS['high_traffic'], '100M')
    add_host_traffic('h12', 's1', COLORS['high_traffic'], '100M')

    # h9(s10) <-> h10(s11)
    add_sw_traffic('s10', 's11', COLORS['high_traffic'], '100M')
    add_host_traffic('h9', 's10', COLORS['high_traffic'], '100M')
    add_host_traffic('h10', 's11', COLORS['high_traffic'], '100M')

    # h17(s10) <-> h15(s11)
    add_sw_traffic('s10', 's11', COLORS['high_traffic'], '100M')
    add_host_traffic('h17', 's10', COLORS['high_traffic'], '100M')
    add_host_traffic('h15', 's11', COLORS['high_traffic'], '100M')

    # h5(s5) <-> h2(s2)
    add_sw_traffic('s5', 's2', COLORS['high_traffic'], '100M')
    add_host_traffic('h5', 's5', COLORS['high_traffic'], '100M')
    add_host_traffic('h2', 's2', COLORS['high_traffic'], '100M')

    # h18(s5) <-> h16(s2)
    add_sw_traffic('s5', 's2', COLORS['high_traffic'], '100M')
    add_host_traffic('h18', 's5', COLORS['high_traffic'], '100M')
    add_host_traffic('h16', 's2', COLORS['high_traffic'], '100M')

    # --- 3M Traffic Pair (Orange) ---
    # h4(s4) <-> h19(s3)
    add_sw_traffic('s4', 's3', COLORS['low_traffic'], '3M')
    add_host_traffic('h4', 's4', COLORS['low_traffic'], '3M')
    add_host_traffic('h19', 's3', COLORS['low_traffic'], '3M')

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    # 1) Draw all switch-to-switch links
    for u, v in all_switch_links:
        key = frozenset([u, v])
        x1, y1 = sw_pos[u]
        x2, y2 = sw_pos[v]

        if key in traffic_on_switch_link:
            entries = traffic_on_switch_link[key]
            colors_present = set(c for c, _ in entries)
            if COLORS['high_traffic'] in colors_present:
                color = COLORS['high_traffic']
                lw = 3.0
            else:
                color = COLORS['low_traffic']
                lw = 2.5
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=3, solid_capstyle='round')
        else:
            ax.plot([x1, x2], [y1, y2], color=COLORS['idle'], linewidth=1.2, zorder=2)

    # 2) Draw all host-to-switch links
    for h_name, (h_coord, sw_name, lbl, sublbl, is_server) in host_info.items():
        hx, hy = h_coord
        sx, sy = sw_pos[sw_name]
        key = (h_name, sw_name)

        if key in traffic_on_host_link:
            entries = traffic_on_host_link[key]
            colors_present = set(c for c, _ in entries)
            if COLORS['high_traffic'] in colors_present:
                color = COLORS['high_traffic']
                lw = 2.5
            else:
                color = COLORS['low_traffic']
                lw = 2.0
            ax.plot([hx, sx], [hy, sy], color=color, linewidth=lw, zorder=3, solid_capstyle='round')
        else:
            ax.plot([hx, sx], [hy, sy], color=COLORS['idle'], linewidth=1.0, zorder=2)

    # 3) Draw Host & Server icons
    for h_name, (h_coord, sw_name, lbl, sublbl, is_server) in host_info.items():
        hx, hy = h_coord
        if is_server:
            draw_server_icon(ax, hx, hy, scale=0.95, label=lbl, sublabel=sublbl)
        elif h_name == 'h0':
            draw_host_icon(ax, hx, hy, scale=0.85, label=lbl, sublabel=sublbl)
        else:
            draw_host_icon(ax, hx, hy, scale=0.72, label=lbl, sublabel=sublbl)

    # 4) Draw Switch icons with s0, s1, s2, ... labels
    for sw_name, (sx, sy) in sw_pos.items():
        draw_switch_icon(ax, sx, sy, scale=1.0, label=sw_name)

    # 5) Draw Floodlight SDN Controller at top center
    draw_controller_icon(ax, 7.8, 7.8, scale=1.1, label="Floodlight SDN Controller")

    # -----------------------------------------------------------------------
    # Traffic Flow Annotations on highlighted links (placed in open spaces)
    # -----------------------------------------------------------------------
    def add_label(x, y, text, color):
        ax.text(x, y, text, fontsize=7.5, fontweight='bold', color=color,
                ha='center', va='center', zorder=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff', edgecolor=color, alpha=0.95, linewidth=0.9))

    # Placed in clear open space directly along each active link:
    add_label(7.2, 5.85, 'h7\u2194h3, h13\u2194h11\n(100M each)', COLORS['high_traffic']) # on s8 <-> s3 link
    add_label(8.0, 4.65, 'h6\u2194h1, h14\u2194h12\n(100M each)', COLORS['high_traffic']) # on s7 <-> s1 link
    add_label(12.7, 5.8, 'h9\u2194h10, h17\u2194h15\n(100M each)', COLORS['high_traffic']) # on s10 <-> s11 link
    add_label(4.1, 1.6, 'h5\u2194h2, h18\u2194h16\n(100M each)', COLORS['high_traffic']) # on s5 <-> s2 link
    add_label(5.4, 3.8, 'h4\u2194h19\n(3M)', COLORS['low_traffic'])                       # on s4 <-> s3 link

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color=COLORS['high_traffic'], lw=3.0, label='High Traffic (100M iperf3)'),
        Line2D([0], [0], color=COLORS['low_traffic'], lw=2.5, label='Low Traffic (3M iperf3)'),
        Line2D([0], [0], color=COLORS['idle'], lw=1.2, label='Idle Link (No Traffic)'),
    ]

    leg = ax.legend(handles=legend_elements, loc='upper left', frameon=True,
                    facecolor='#ffffff', edgecolor='#333333', fontsize=10, shadow=True,
                    bbox_to_anchor=(-0.02, 1.02))
    leg.set_zorder(10)

    ax.set_title("NSFNET SDN Topology \u2014 Traffic Flow Visualization",
                 fontsize=16, fontweight='bold', pad=15)

    ax.set_xlim(-1.2, 16.8)
    ax.set_ylim(-1.2, 9.6)
    ax.axis('off')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Traffic topology image generated at: {output_path}")


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic_topology.png")
    generate_traffic_topology(output_path)
