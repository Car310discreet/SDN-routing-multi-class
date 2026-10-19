#!/usr/bin/env python3
"""
2D NSFNET SDN Topology Generator Script
Generates a publication-grade, monochrome schematic topology diagram matching
the visual style and icon design of SDN literature figures.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set MPLCONFIGDIR to a writable temporary directory
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

def draw_switch_icon(ax, x, y, scale=1.0, label=""):
    """Draws a 1U Rack-mount Network Switch vector icon."""
    w = 0.65 * scale
    h = 0.22 * scale
    
    # Outer switch chassis (black body)
    chassis = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor='#141414', edgecolor='#000000', linewidth=1.2, zorder=5
    )
    ax.add_patch(chassis)
    
    # Rack ears (left & right mounting flanges)
    ear_w = 0.04 * scale
    ear_h = 0.16 * scale
    ax.add_patch(patches.Rectangle((x - w/2 - ear_w, y - ear_h/2), ear_w, ear_h, facecolor='#333333', edgecolor='#000000', linewidth=0.8, zorder=4))
    ax.add_patch(patches.Rectangle((x + w/2, y - ear_h/2), ear_w, ear_h, facecolor='#333333', edgecolor='#000000', linewidth=0.8, zorder=4))
    
    # Row of Ethernet RJ45 Ports (white squares)
    num_ports = 12
    port_w = (w * 0.68) / num_ports
    port_h = h * 0.38
    start_x = x - (w * 0.34) + port_w/2
    port_y = y - port_h/2
    
    for i in range(num_ports):
        px = start_x + i * port_w
        p = patches.Rectangle((px - port_w*0.4, port_y), port_w*0.78, port_h, facecolor='#ffffff', edgecolor='#222222', linewidth=0.4, zorder=6)
        ax.add_patch(p)
        
    # Status LED indicators on far left
    led_x = x - w/2 + 0.06 * scale
    ax.plot(led_x, y + 0.03*scale, marker='o', markersize=2.5*scale, color='#ffffff', zorder=6)
    ax.plot(led_x, y - 0.03*scale, marker='o', markersize=2.5*scale, color='#ffffff', zorder=6)
    
    # Switch label centered above the icon (e.g. s0, s1, ...)
    if label:
        ax.text(x, y + h/2 + 0.12*scale, label, fontsize=12*scale, fontweight='bold', color='#000000', ha='center', va='bottom', zorder=7)

def draw_host_icon(ax, x, y, scale=1.0, label="", sublabel=""):
    """Draws a Desktop PC / Host vector icon."""
    mw = 0.42 * scale
    mh = 0.30 * scale
    my = y + 0.06 * scale
    
    # Outer Monitor Frame
    mon = patches.FancyBboxPatch(
        (x - mw/2, my - mh/2), mw, mh,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor='#1a1a1a', edgecolor='#000000', linewidth=1.2, zorder=5
    )
    ax.add_patch(mon)
    
    # Display Screen Area
    sw = mw * 0.82
    sh = mh * 0.75
    screen = patches.Rectangle((x - sw/2, my - sh/2), sw, sh, facecolor='#ffffff', edgecolor='#444444', linewidth=0.6, zorder=6)
    ax.add_patch(screen)
    
    # Screen inner graphic lines
    ax.plot([x - sw*0.35, x + sw*0.35], [my + sh*0.12, my + sh*0.12], color='#222222', lw=1.0*scale, zorder=7)
    ax.plot([x - sw*0.35, x + sw*0.05], [my - sh*0.15, my - sh*0.15], color='#666666', lw=1.0*scale, zorder=7)
    
    # Stand Neck & Base
    stand_w = 0.06 * scale
    stand_h = 0.08 * scale
    stand_y = my - mh/2 - stand_h
    ax.add_patch(patches.Rectangle((x - stand_w/2, stand_y), stand_w, stand_h, facecolor='#333333', edgecolor='#000000', linewidth=0.6, zorder=4))
    
    base_w = 0.24 * scale
    base_h = 0.03 * scale
    ax.add_patch(patches.Rectangle((x - base_w/2, stand_y - base_h), base_w, base_h, facecolor='#1a1a1a', edgecolor='#000000', linewidth=0.6, zorder=4))
    
    # Labels below icon (e.g. h0, h1, ...)
    if label:
        ax.text(x, y - 0.22*scale, label, fontsize=10.5*scale, fontweight='bold', color='#000000', ha='center', va='top', zorder=7)
    if sublabel:
        ax.text(x, y - 0.36*scale, sublabel, fontsize=8.5*scale, fontweight='bold', color='#444444', ha='center', va='top', zorder=7)

def draw_server_icon(ax, x, y, scale=1.0, label="", sublabel=""):
    """Draws a Vertical Enterprise Server Tower vector icon."""
    w = 0.32 * scale
    h = 0.52 * scale
    
    # Outer Chassis
    chassis = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor='#141414', edgecolor='#000000', linewidth=1.2, zorder=5
    )
    ax.add_patch(chassis)
    
    # Drive Bays / Slots
    num_bays = 4
    bay_h = (h * 0.65) / num_bays
    bay_w = w * 0.78
    start_y = y + (h * 0.30) - bay_h/2
    
    for i in range(num_bays):
        by = start_y - i * (bay_h * 1.1)
        bay = patches.Rectangle((x - bay_w/2, by), bay_w, bay_h, facecolor='#3a3a3a', edgecolor='#000000', linewidth=0.6, zorder=6)
        ax.add_patch(bay)
        # Drive handle line
        ax.plot([x - bay_w*0.35, x + bay_w*0.1], [by + bay_h/2, by + bay_h/2], color='#ffffff', lw=0.8*scale, zorder=7)
        # Status LED
        ax.plot(x + bay_w*0.3, by + bay_h/2, marker='s', markersize=2.2*scale, color='#ffffff', zorder=7)
        
    # Power Button
    ax.plot(x, y - h*0.38, marker='o', markersize=4.0*scale, color='#ffffff', zorder=7)
    
    if label:
        ax.text(x, y - h/2 - 0.08*scale, label, fontsize=10.5*scale, fontweight='bold', color='#000000', ha='center', va='top', zorder=7)
    if sublabel:
        ax.text(x, y - h/2 - 0.22*scale, sublabel, fontsize=8.5*scale, fontweight='bold', color='#444444', ha='center', va='top', zorder=7)

def draw_controller_icon(ax, x, y, scale=1.0, label="Floodlight SDN Controller"):
    """Draws an SDN Controller box icon with equalizer control sliders."""
    w = 0.60 * scale
    h = 0.38 * scale
    
    box = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor='#ffffff', edgecolor='#000000', linewidth=1.4, zorder=5
    )
    ax.add_patch(box)
    
    # SDN Title Banner
    hdr_h = 0.10 * scale
    hdr = patches.Rectangle((x - w/2 + 0.02*scale, y + h/2 - hdr_h - 0.02*scale), w - 0.04*scale, hdr_h, facecolor='#000000', zorder=6)
    ax.add_patch(hdr)
    ax.text(x, y + h/2 - hdr_h/2 - 0.02*scale, "SDN", color='#ffffff', fontweight='bold', fontsize=9*scale, ha='center', va='center', zorder=7)
    
    # Equalizer Sliders / Controls
    num_sliders = 3
    slider_x_start = x - w*0.25
    slider_dx = w*0.25
    for i in range(num_sliders):
        sx = slider_x_start + i*slider_dx
        sy_top = y + h*0.06
        sy_bot = y - h*0.36
        ax.plot([sx, sx], [sy_bot, sy_top], color='#000000', lw=1.2*scale, zorder=6)
        knob_y = y - h*0.18 + i*0.06*scale
        ax.add_patch(patches.Rectangle((sx - 0.04*scale, knob_y - 0.025*scale), 0.08*scale, 0.05*scale, facecolor='#000000', zorder=7))
        
    if label:
        ax.text(x, y + h/2 + 0.08*scale, label, fontsize=12*scale, fontweight='bold', color='#000000', ha='center', va='bottom', zorder=7)

def generate_topology_diagram(output_path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_facecolor('#ffffff')

    # 14 NSFNET Switch Coordinates matching reference diagram
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

    # Inter-Switch Connections (Solid Black Backbone Links)
    switch_links = [
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

    # Draw Inter-Switch Links
    for u, v in switch_links:
        x1, y1 = sw_pos[u]
        x2, y2 = sw_pos[v]
        ax.plot([x1, x2], [y1, y2], color='#000000', linewidth=1.5, zorder=2)

    # Host attachments for all 20 hosts (h0-h19) in project
    # (coord, attached_switch, label, sublabel, is_server)
    host_pos = {
        # s0
        'h0':  ((-0.5, 4.0), 's0', 'h0', '(Receiver)', False),
        # s1
        'h1':  ((2.2, 7.6), 's1', 'h1', '', False),
        'h12': ((3.4, 7.6), 's1', 'h12', '', False),
        # s2
        'h2':  ((1.8, 0.7), 's2', 'h2', '', False),
        'h16': ((3.2, 0.7), 's2', 'h16', '', False),
        # s3
        'h3':  ((4.0, 5.7), 's3', 'h3', '', False),
        'h11': ((5.2, 5.7), 's3', 'h11', '', False),
        'h19': ((3.8, 3.2), 's3', 'h19', '', False),
        # s4
        'h4':  ((6.5, 2.2), 's4', 'h4', '', False),
        # s5
        'h5':  ((4.5, -0.4), 's5', 'h5', '', False),
        'h18': ((6.3, -0.4), 's5', 'h18', '', False),
        # s7
        'h6':  ((8.4, 5.8), 's7', 'h6', '', False),
        'h14': ((9.4, 5.8), 's7', 'h14', '', False),
        # s8
        'h7':  ((9.5, 7.9), 's8', 'h7', '', False),
        'h13': ((10.7, 7.9), 's8', 'h13', '', False),
        # s9
        'h8':  ((15.5, 4.8), 's9', 'h8', '(Sender)', True),
        # s10
        'h9':  ((11.4, 3.2), 's10', 'h9', '', False),
        'h17': ((12.5, 3.2), 's10', 'h17', '', False),
        # s11
        'h10': ((11.4, 8.8), 's11', 'h10', '', False),
        'h15': ((12.8, 8.8), 's11', 'h15', '', False),
    }

    # Draw Host-to-Switch Links
    for h_name, (h_coord, sw_name, lbl, sublbl, is_server) in host_pos.items():
        hx, hy = h_coord
        sx, sy = sw_pos[sw_name]
        ax.plot([hx, sx], [hy, sy], color='#000000', linewidth=1.2, zorder=2)

    # Draw Host & Server Icons
    for h_name, (h_coord, sw_name, lbl, sublbl, is_server) in host_pos.items():
        hx, hy = h_coord
        if is_server:
            draw_server_icon(ax, hx, hy, scale=0.95, label=lbl, sublabel=sublbl)
        elif h_name == 'h0':
            draw_host_icon(ax, hx, hy, scale=0.85, label=lbl, sublabel=sublbl)
        else:
            draw_host_icon(ax, hx, hy, scale=0.72, label=lbl, sublabel=sublbl)

    # Draw Switch Icons with s0, s1, s2, ... labels
    for sw_name, (sx, sy) in sw_pos.items():
        draw_switch_icon(ax, sx, sy, scale=1.0, label=sw_name)

    # Draw Floodlight SDN Controller at Top Center
    draw_controller_icon(ax, 7.8, 7.8, scale=1.1, label="Floodlight SDN Controller")

    # Legend in Top-Left Corner (clean & unclipped)
    leg_x, leg_y = -0.8, 8.8
    
    # Legend Item 1: Link
    ax.plot([leg_x, leg_x + 0.5], [leg_y, leg_y], color='#000000', lw=1.8, zorder=9)
    ax.text(leg_x + 0.65, leg_y, "Link", fontsize=11, fontweight='bold', color='#000000', va='center', zorder=9)
    
    # Legend Item 2: Host
    draw_host_icon(ax, leg_x + 0.25, leg_y - 0.45, scale=0.45)
    ax.text(leg_x + 0.65, leg_y - 0.45, "Host", fontsize=11, fontweight='bold', color='#000000', va='center', zorder=9)
    
    # Legend Item 3: Switch
    draw_switch_icon(ax, leg_x + 0.25, leg_y - 0.90, scale=0.55)
    ax.text(leg_x + 0.65, leg_y - 0.90, "Switch", fontsize=11, fontweight='bold', color='#000000', va='center', zorder=9)
    
    # Legend Item 4: Server
    draw_server_icon(ax, leg_x + 0.25, leg_y - 1.40, scale=0.42)
    ax.text(leg_x + 0.65, leg_y - 1.40, "Server", fontsize=11, fontweight='bold', color='#000000', va='center', zorder=9)

    ax.set_xlim(-1.2, 16.8)
    ax.set_ylim(-1.2, 9.6)
    ax.axis('off')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Topography successfully generated at: {output_path}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topography.png")
    generate_topology_diagram(output_path)
