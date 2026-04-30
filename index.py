import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

class ZeemanSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Zeeman Effect Simulator - An Interactive Study of Magnetic Field Effects on Atomic Spectra")
        self.root.geometry("1600x1000")
        
        # Dark theme colors
        self.bg_color = '#000000'
        self.fg_color = '#FFFFFF'
        self.panel_bg = '#1a1a1a'
        self.accent_color = '#00aaff'
        
        # Configure root background
        self.root.configure(bg=self.bg_color)
        
        # Configure ttk style
        self.setup_dark_theme()
        
        # Element data
        self.elements = [
            {'value': 'hydrogen', 'label': 'Hydrogen (H)', 'symbol': 'H', 'wavelength': 656.3, 'line': 'H-α', 'color': '#ef4444'},
            {'value': 'helium', 'label': 'Helium (He)', 'symbol': 'He', 'wavelength': 587.6, 'line': 'D₃', 'color': '#f59e0b'},
            {'value': 'sodium', 'label': 'Sodium (Na)', 'symbol': 'Na', 'wavelength': 589.3, 'line': 'D-line', 'color': '#fbbf24'},
            {'value': 'mercury', 'label': 'Mercury (Hg)', 'symbol': 'Hg', 'wavelength': 546.1, 'line': 'Green', 'color': '#22c55e'},
            {'value': 'cadmium', 'label': 'Cadmium (Cd)', 'symbol': 'Cd', 'wavelength': 643.8, 'line': 'Red', 'color': '#dc2626'},
            {'value': 'zinc', 'label': 'Zinc (Zn)', 'symbol': 'Zn', 'wavelength': 468.0, 'line': 'Blue', 'color': '#3b82f6'}
        ]
        
        # State variables
        self.magnetic_field = tk.DoubleVar(value=0.0)
        self.zeeman_type = tk.StringVar(value='normal')
        self.selected_element = tk.StringVar(value='hydrogen')
        self.show_sigma_minus = tk.BooleanVar(value=True)
        self.show_pi = tk.BooleanVar(value=True)
        self.show_sigma_plus = tk.BooleanVar(value=True)
        
        self.bohr_magneton = 5.788e-5  # eV/T
        
        # Animation variables
        self.is_running = False
        self.animation_speed = tk.DoubleVar(value=0.1)
        self.animation_direction = 1
        
        self.create_widgets()
        self.update_plots()
    
    def setup_dark_theme(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        style.configure('TLabelframe', background=self.bg_color, foreground=self.accent_color, bordercolor=self.accent_color)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.accent_color, font=('Arial', 10, 'bold'))
        style.configure('TButton', background=self.panel_bg, foreground=self.fg_color)
        style.configure('TCheckbutton', background=self.bg_color, foreground=self.fg_color)
        style.configure('TRadiobutton', background=self.bg_color, foreground=self.fg_color)
        style.configure('TCombobox', fieldbackground=self.panel_bg, background=self.panel_bg, foreground=self.fg_color)
        
        # Configure hover effects
        style.map('TCheckbutton', background=[('active', self.bg_color)])
        style.map('TRadiobutton', background=[('active', self.bg_color)])
    
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="⚛ ZEEMAN EFFECT SIMULATOR", 
                               font=('Arial', 22, 'bold'), foreground=self.accent_color)
        title_label.grid(row=0, column=0, columnspan=3, pady=(10, 2))
        
        # Subtitle
        subtitle_label = ttk.Label(main_frame, 
                                   text="AN INTERACTIVE STUDY OF MAGNETIC FIELD EFFECTS ON ATOMIC SPECTRA", 
                                   font=('Arial', 11), foreground='#66ccff')
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls & Settings", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Element selection
        ttk.Label(control_frame, text="Element:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=5)
        
        element_combo = ttk.Combobox(control_frame, textvariable=self.selected_element,
                                     values=[e['value'] for e in self.elements],
                                     state='readonly', width=20)
        element_combo.grid(row=1, column=0, sticky=tk.W, pady=5)
        element_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plots())
        
        # Element info
        self.element_info = ttk.Label(control_frame, text="", foreground=self.accent_color, font=('Arial', 9))
        self.element_info.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        # Magnetic field slider
        ttk.Label(control_frame, text="Magnetic Field Strength (Tesla):", 
                 font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky=tk.W, pady=(15, 5))
        
        field_frame = ttk.Frame(control_frame)
        field_frame.grid(row=4, column=0, sticky=tk.W, pady=5)
        
        self.field_slider = ttk.Scale(field_frame, from_=0, to=10, 
                                     variable=self.magnetic_field,
                                     orient=tk.HORIZONTAL, length=300,
                                     command=lambda v: self.update_plots())
        self.field_slider.grid(row=0, column=0)
        
        self.field_label = ttk.Label(field_frame, text="0.0 T", foreground=self.accent_color, 
                                    font=('Arial', 10, 'bold'))
        self.field_label.grid(row=0, column=1, padx=10)
        
        # Zeeman type selection
        ttk.Label(control_frame, text="Zeeman Effect Type:", 
                 font=('Arial', 10, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=(15, 5))
        
        ttk.Radiobutton(control_frame, text="Normal (¹S → ¹P)", 
                       variable=self.zeeman_type, value='normal',
                       command=self.update_plots).grid(row=6, column=0, sticky=tk.W)
        
        ttk.Radiobutton(control_frame, text="Anomalous (²S → ²P)", 
                       variable=self.zeeman_type, value='anomalous',
                       command=self.update_plots).grid(row=7, column=0, sticky=tk.W)
        
        # Transition selection
        ttk.Label(control_frame, text="Transition Selection:", 
                 font=('Arial', 10, 'bold')).grid(row=8, column=0, sticky=tk.W, pady=(15, 5))
        
        ttk.Checkbutton(control_frame, text="σ⁻ (Δm = -1, Left circular)", 
                       variable=self.show_sigma_minus,
                       command=self.update_plots).grid(row=9, column=0, sticky=tk.W)
        
        ttk.Checkbutton(control_frame, text="π (Δm = 0, Linear)", 
                       variable=self.show_pi,
                       command=self.update_plots).grid(row=10, column=0, sticky=tk.W)
        
        ttk.Checkbutton(control_frame, text="σ⁺ (Δm = +1, Right circular)", 
                       variable=self.show_sigma_plus,
                       command=self.update_plots).grid(row=11, column=0, sticky=tk.W)
        
        # Animation controls
        ttk.Label(control_frame, text="Animation Controls:", 
                 font=('Arial', 10, 'bold')).grid(row=12, column=0, sticky=tk.W, pady=(15, 5))
        
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=13, column=0, sticky=tk.W, pady=5)
        
        self.run_button = tk.Button(button_frame, text="▶ Run Simulation", 
                                    command=self.toggle_animation,
                                    bg='#00aa00', fg='white', 
                                    font=('Arial', 10, 'bold'),
                                    relief=tk.RAISED, bd=3,
                                    padx=20, pady=8,
                                    cursor='hand2')
        self.run_button.grid(row=0, column=0, padx=5)
        
        reset_button = tk.Button(button_frame, text="⟲ Reset", 
                                command=self.reset_simulation,
                                bg='#ff6600', fg='white', 
                                font=('Arial', 10, 'bold'),
                                relief=tk.RAISED, bd=3,
                                padx=20, pady=8,
                                cursor='hand2')
        reset_button.grid(row=0, column=1, padx=5)
        
        # Animation speed
        ttk.Label(control_frame, text="Animation Speed:", 
                 font=('Arial', 9)).grid(row=14, column=0, sticky=tk.W, pady=(10, 2))
        
        speed_frame = ttk.Frame(control_frame)
        speed_frame.grid(row=15, column=0, sticky=tk.W, pady=5)
        
        ttk.Scale(speed_frame, from_=0.01, to=0.5, 
                 variable=self.animation_speed,
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=0)
        
        ttk.Label(speed_frame, text="Slow", font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(speed_frame, text="Fast", font=('Arial', 8)).grid(row=1, column=0, sticky=tk.E)
        
        # Physics info
        info_frame = ttk.LabelFrame(control_frame, text="Key Equations", padding="10")
        info_frame.grid(row=16, column=0, sticky=(tk.W, tk.E), pady=10)
        
        equations = [
            "Energy shift: ΔE = gⱼ·mⱼ·μB·B",
            "Wavelength shift: Δλ = (λ²/hc)·ΔE",
            "Bohr magneton: μB = 5.788×10⁻⁵ eV/T",
            "Landé g-factor: gⱼ = 1 + [J(J+1)+S(S+1)-L(L+1)]/[2J(J+1)]",
            "Selection rules: ΔJ = 0,±1; Δmⱼ = 0,±1"
        ]
        
        for i, eq in enumerate(equations):
            ttk.Label(info_frame, text=eq, font=('Courier', 8), foreground='#00ff00').grid(
                row=i, column=0, sticky=tk.W, pady=2)
        
        # Middle panel - Plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure with dark background
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 16))
        self.fig.patch.set_facecolor('#000000')
        
        # Add spacing between subplots
        self.fig.subplots_adjust(hspace=0.4, top=0.97, bottom=0.05, left=0.1, right=0.95)
        self.fig.tight_layout(pad=4.0, h_pad=3.0)  # Increased padding between plots
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def insert_physics_theory(self):
        # Theory section removed
        pass
        
        # Add color tags
        self.theory_text.tag_config('header', foreground='#00aaff', font=('Arial', 10, 'bold'))
        self.theory_text.tag_config('equation', foreground='#00ff00', font=('Courier', 9))
        self.theory_text.tag_config('important', foreground='#ff6666', font=('Arial', 9, 'bold'))
    
    def get_current_element(self):
        element_value = self.selected_element.get()
        return next(e for e in self.elements if e['value'] == element_value)
    
    def calculate_energy_splitting(self):
        B = self.magnetic_field.get()
        muB = self.bohr_magneton
        zeeman_type = self.zeeman_type.get()
        
        if zeeman_type == 'normal':
            upper_levels = [{'m': m, 'shift': m * muB * B * 1000, 
                           'label': f'm = {m:+d}' if m != 0 else 'm = 0'} 
                          for m in [-1, 0, 1]]
            lower_levels = [{'m': 0, 'shift': 0, 'label': 'm = 0'}]
        else:
            g_upper = 1.5
            g_lower = 1.0
            upper_levels = [{'m': m, 'shift': m * g_upper * muB * B * 1000, 
                           'label': f'mⱼ = {m:+d}' if m != 0 else 'mⱼ = 0',
                           'g': g_upper} 
                          for m in [-1, 0, 1]]
            lower_levels = [{'m': m, 'shift': m * g_lower * muB * B * 1000, 
                           'label': f'mⱼ = {m:+d}' if m != 0 else 'mⱼ = 0',
                           'g': g_lower} 
                          for m in [-1, 0, 1]]
        
        return upper_levels, lower_levels
    
    def calculate_spectral_lines(self):
        element = self.get_current_element()
        base_wavelength = element['wavelength']
        B = self.magnetic_field.get()
        zeeman_type = self.zeeman_type.get()
        
        lines = []
        
        if zeeman_type == 'normal':
            if self.show_sigma_minus.get():
                lines.append({
                    'wavelength': base_wavelength - B * 0.05,
                    'intensity': 0.8,
                    'polarization': 'σ⁻',
                    'color': '#3b82f6',
                    'transition': 'Δm = -1'
                })
            
            if self.show_pi.get():
                lines.append({
                    'wavelength': base_wavelength,
                    'intensity': 1.0,
                    'polarization': 'π',
                    'color': '#22c55e',
                    'transition': 'Δm = 0'
                })
            
            if self.show_sigma_plus.get():
                lines.append({
                    'wavelength': base_wavelength + B * 0.05,
                    'intensity': 0.8,
                    'polarization': 'σ⁺',
                    'color': '#ef4444',
                    'transition': 'Δm = +1'
                })
        else:
            transitions = [
                {'dm': -1, 'upper': -1, 'lower': 0, 'show': self.show_sigma_minus.get()},
                {'dm': -1, 'upper': 0, 'lower': 1, 'show': self.show_sigma_minus.get()},
                {'dm': 0, 'upper': -1, 'lower': -1, 'show': self.show_pi.get()},
                {'dm': 0, 'upper': 0, 'lower': 0, 'show': self.show_pi.get()},
                {'dm': 0, 'upper': 1, 'lower': 1, 'show': self.show_pi.get()},
                {'dm': 1, 'upper': 0, 'lower': -1, 'show': self.show_sigma_plus.get()},
                {'dm': 1, 'upper': 1, 'lower': 0, 'show': self.show_sigma_plus.get()},
            ]
            
            for t in transitions:
                if t['show']:
                    shift = (t['upper'] * 1.5 - t['lower'] * 1.0) * B * 0.03
                    lines.append({
                        'wavelength': base_wavelength + shift,
                        'intensity': 0.7 + np.random.random() * 0.3,
                        'polarization': 'π' if t['dm'] == 0 else ('σ⁺' if t['dm'] > 0 else 'σ⁻'),
                        'color': '#22c55e' if t['dm'] == 0 else ('#ef4444' if t['dm'] > 0 else '#3b82f6'),
                        'transition': f"Δm = {t['dm']:+d}" if t['dm'] != 0 else "Δm = 0"
                    })
        
        return lines
    
    def plot_energy_levels(self):
        self.ax1.clear()
        self.ax1.set_facecolor('#000000')
        
        upper_levels, lower_levels = self.calculate_energy_splitting()
        B = self.magnetic_field.get()
        zeeman_type = self.zeeman_type.get()
        
        # Calculate actual energy range
        all_shifts = [l['shift'] for l in upper_levels + lower_levels]
        max_shift = max([abs(s) for s in all_shifts]) if all_shifts else 1
        
        # Set up plot with proper energy scaling
        self.ax1.set_ylim(-max_shift*1.3, max_shift*1.3)
        self.ax1.set_xlim(-2, 3.5)
        
        self.ax1.set_ylabel('Energy Shift (meV)', fontsize=12, fontweight='bold', color='white')
        self.ax1.set_title(f'Zeeman Energy Level Splitting - {"¹S → ¹P" if zeeman_type == "normal" else "²S → ²P"} Transition', 
                          fontsize=14, fontweight='bold', color='#00aaff')
        
        # Draw zero energy reference line
        self.ax1.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=1, label='B = 0')
        
        # Define positions for B=0 and B≠0 states
        x_before = 0.5   # Position at B=0
        x_after_upper = 2.0   # Position at B≠0 for upper state
        x_after_lower = 2.0   # Position at B≠0 for lower state
        line_width = 0.4
        
        # Determine if states split at B=0 (for lower state in anomalous)
        upper_splits = len(upper_levels) > 1
        lower_splits = len(lower_levels) > 1
        
        # Draw UPPER state
        # Before field (B=0) - single level or degenerate
        upper_energy_before = upper_levels[0]['shift'] if len(upper_levels) == 1 else 0
        
        if not upper_splits or B == 0:
            # Single line at B=0
            self.ax1.plot([x_before - line_width/2, x_before + line_width/2], 
                         [upper_energy_before, upper_energy_before], 
                         'cyan', linewidth=4, solid_capstyle='butt', label='Upper State (B=0)')
            self.ax1.text(x_before - 0.7, upper_energy_before, 'Upper\nState', 
                         ha='right', va='center', fontsize=10, color='cyan', fontweight='bold')
        
        # After field (B≠0) - split levels
        if B > 0:
            for i, level in enumerate(upper_levels):
                energy = level['shift']
                # Draw energy level line
                self.ax1.plot([x_after_upper - line_width/2, x_after_upper + line_width/2], 
                             [energy, energy], 
                             'cyan', linewidth=3, solid_capstyle='butt', alpha=0.9)
                
                # Label with quantum number
                self.ax1.text(x_after_upper + line_width/2 + 0.05, energy, 
                             level['label'], 
                             ha='left', va='center', fontsize=9, color='cyan', fontweight='bold')
                
                # Energy value
                self.ax1.text(x_after_upper + line_width/2 + 0.6, energy, 
                             f"{energy:.3f} meV", 
                             ha='left', va='center', fontsize=7, color='cyan', alpha=0.7)
                
                # Draw splitting lines from B=0 to B≠0
                if upper_splits:
                    self.ax1.plot([x_before + line_width/2, x_after_upper - line_width/2], 
                                 [upper_energy_before, energy], 
                                 'cyan', linewidth=1, linestyle='--', alpha=0.4)
        
        # Draw LOWER state
        # Before field (B=0)
        lower_energy_before = lower_levels[0]['shift'] if len(lower_levels) == 1 else 0
        
        if not lower_splits or B == 0:
            # Single line at B=0
            self.ax1.plot([x_before - line_width/2, x_before + line_width/2], 
                         [lower_energy_before, lower_energy_before], 
                         'orange', linewidth=4, solid_capstyle='butt', label='Lower State (B=0)')
            self.ax1.text(x_before - 0.7, lower_energy_before, 'Lower\nState', 
                         ha='right', va='center', fontsize=10, color='orange', fontweight='bold')
        
        # After field (B≠0) - split levels
        if B > 0:
            for i, level in enumerate(lower_levels):
                energy = level['shift']
                # Draw energy level line
                self.ax1.plot([x_after_lower - line_width/2, x_after_lower + line_width/2], 
                             [energy, energy], 
                             'orange', linewidth=3, solid_capstyle='butt', alpha=0.9)
                
                # Label with quantum number
                self.ax1.text(x_after_lower + line_width/2 + 0.05, energy, 
                             level['label'], 
                             ha='left', va='center', fontsize=9, color='orange', fontweight='bold')
                
                # Energy value
                self.ax1.text(x_after_lower + line_width/2 + 0.6, energy, 
                             f"{energy:.3f} meV", 
                             ha='left', va='center', fontsize=7, color='orange', alpha=0.7)
                
                # Draw splitting lines from B=0 to B≠0
                if lower_splits:
                    self.ax1.plot([x_before + line_width/2, x_after_lower - line_width/2], 
                                 [lower_energy_before, energy], 
                                 'orange', linewidth=1, linestyle='--', alpha=0.4)
        
        # Draw allowed transitions (only if B > 0)
        if B > 0:
            for upper in upper_levels:
                for lower in lower_levels:
                    dm = upper['m'] - lower['m']
                    
                    # Check selection rules
                    if abs(dm) > 1:
                        continue
                    
                    # Check which transitions to show
                    show = False
                    color = '#22c55e'
                    alpha = 0.5
                    
                    if dm == -1 and self.show_sigma_minus.get():
                        show, color, alpha = True, '#3b82f6', 0.6
                    elif dm == 0 and self.show_pi.get():
                        show, color, alpha = True, '#22c55e', 0.6
                    elif dm == 1 and self.show_sigma_plus.get():
                        show, color, alpha = True, '#ef4444', 0.6
                    
                    if show:
                        # Draw transition arrow
                        arrow = FancyArrowPatch(
                            (x_after_upper, upper['shift']), 
                            (x_after_lower, lower['shift']),
                            arrowstyle='->', 
                            mutation_scale=20,
                            color=color, 
                            alpha=alpha, 
                            linewidth=2,
                            zorder=1
                        )
                        self.ax1.add_patch(arrow)
        
        # Add labels for B=0 and B≠0 regions
        if B > 0:
            self.ax1.text(x_before, max_shift*1.15, 'B = 0', 
                         ha='center', va='bottom', fontsize=11, color='white',
                         bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.7))
            self.ax1.text(x_after_upper, max_shift*1.15, f'B = {B:.2f} T', 
                         ha='center', va='bottom', fontsize=11, color='white',
                         bbox=dict(boxstyle='round', facecolor='#00aaff', alpha=0.7))
        
        # Add legend for transitions
        if B > 0:
            from matplotlib.patches import Patch
            legend_elements = []
            if self.show_sigma_minus.get():
                legend_elements.append(Patch(facecolor='#3b82f6', label='σ⁻ (Δm=-1)'))
            if self.show_pi.get():
                legend_elements.append(Patch(facecolor='#22c55e', label='π (Δm=0)'))
            if self.show_sigma_plus.get():
                legend_elements.append(Patch(facecolor='#ef4444', label='σ⁺ (Δm=+1)'))
            
            if legend_elements:
                self.ax1.legend(handles=legend_elements, loc='lower right', fontsize=9,
                               facecolor='#1a1a1a', edgecolor='#00aaff', framealpha=0.9)
        
        # Add info box
        info_text = f"Zeeman Type: {zeeman_type.capitalize()}\n"
        if B > 0:
            info_text += f"Upper levels: {len(upper_levels)}\n"
            info_text += f"Lower levels: {len(lower_levels)}\n"
            if max_shift > 0:
                info_text += f"Max splitting: ±{max_shift:.3f} meV"
        else:
            info_text += "No field applied\nIncrease B to see splitting"
        
        self.ax1.text(0.02, 0.98, info_text, transform=self.ax1.transAxes,
                     fontsize=9, ha='left', va='top', color='yellow',
                     bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8),
                     fontweight='bold')
        
        # Styling
        self.ax1.set_xticks([])
        self.ax1.grid(True, alpha=0.15, color='#444444', axis='y')
        self.ax1.spines['bottom'].set_color('#444444')
        self.ax1.spines['top'].set_color('#444444')
        self.ax1.spines['left'].set_color('#444444')
        self.ax1.spines['right'].set_color('#444444')
        self.ax1.tick_params(colors='white')
    
    def plot_spectrum(self):
        self.ax2.clear()
        self.ax2.set_facecolor('#000000')
        
        element = self.get_current_element()
        lines = self.calculate_spectral_lines()
        base_wavelength = element['wavelength']
        B = self.magnetic_field.get()
        
        # Dynamically calculate wavelength range based on actual lines
        if len(lines) > 0:
            all_wavelengths = [l['wavelength'] for l in lines]
            min_wavelength = min(all_wavelengths + [base_wavelength]) - 0.5
            max_wavelength = max(all_wavelengths + [base_wavelength]) + 0.5
        else:
            min_wavelength = base_wavelength - 3
            max_wavelength = base_wavelength + 3
        
        # Set up plot
        self.ax2.set_xlim(min_wavelength, max_wavelength)
        self.ax2.set_ylim(0, 1.2)
        self.ax2.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold', color='white')
        self.ax2.set_ylabel('Intensity', fontsize=12, fontweight='bold', color='white')
        self.ax2.set_title(f'Emission Spectrum - {element["symbol"]} {element["line"]} ({element["wavelength"]} nm)', 
                          fontsize=14, fontweight='bold', color='#00aaff')
        
        # Draw base line (B=0)
        self.ax2.axvline(x=base_wavelength, color='#666666', linestyle='--', 
                        alpha=0.5, linewidth=2, label=f'{base_wavelength} nm (B=0)')
        
        # Draw spectral lines
        for line in lines:
            color = line['color']
            self.ax2.plot([line['wavelength'], line['wavelength']], 
                         [0, line['intensity']],
                         color=color, linewidth=4, alpha=0.9)
            
            # Add Gaussian profile
            x = np.linspace(line['wavelength'] - 0.15, line['wavelength'] + 0.15, 100)
            y = line['intensity'] * np.exp(-((x - line['wavelength'])/0.04)**2)
            self.ax2.fill_between(x, 0, y, color=color, alpha=0.4)
            
            # Add wavelength label on each peak
            self.ax2.text(line['wavelength'], line['intensity'] + 0.05, 
                         f'{line["wavelength"]:.2f}', 
                         ha='center', va='bottom', fontsize=8, color=color,
                         fontweight='bold', rotation=0)
        
        # Legend
        if len(lines) > 0:
            # Create custom legend entries
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#666666', alpha=0.5, label=f'B=0: {base_wavelength} nm')]
            
            # Group lines by polarization
            sigma_minus = [l for l in lines if l['polarization'] == 'σ⁻']
            pi = [l for l in lines if l['polarization'] == 'π']
            sigma_plus = [l for l in lines if l['polarization'] == 'σ⁺']
            
            if sigma_minus:
                legend_elements.append(Patch(facecolor='#3b82f6', label=f'σ⁻ ({len(sigma_minus)} lines)'))
            if pi:
                legend_elements.append(Patch(facecolor='#22c55e', label=f'π ({len(pi)} lines)'))
            if sigma_plus:
                legend_elements.append(Patch(facecolor='#ef4444', label=f'σ⁺ ({len(sigma_plus)} lines)'))
            
            self.ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                           facecolor='#1a1a1a', edgecolor='#00aaff', framealpha=0.9)
        
        # Add info box showing splitting
        if len(lines) > 1:
            wavelengths = [l['wavelength'] for l in lines]
            total_split = max(wavelengths) - min(wavelengths)
            info_text = f'B = {B:.1f} T\nTotal Δλ = {total_split:.3f} nm\n{len(lines)} lines'
            self.ax2.text(0.02, 0.97, info_text, transform=self.ax2.transAxes,
                         fontsize=10, ha='left', va='top', color='yellow',
                         bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8),
                         fontweight='bold')
        
        self.ax2.grid(True, alpha=0.2, color='#444444')
        self.ax2.spines['bottom'].set_color('#444444')
        self.ax2.spines['top'].set_color('#444444')
        self.ax2.spines['left'].set_color('#444444')
        self.ax2.spines['right'].set_color('#444444')
        self.ax2.tick_params(colors='white')
    
    def plot_energy_vs_field(self):
        """Plot energy levels as a function of magnetic field strength"""
        self.ax3.clear()
        self.ax3.set_facecolor('#000000')
        
        zeeman_type = self.zeeman_type.get()
        current_B = self.magnetic_field.get()
        
        # Generate field range
        B_range = np.linspace(0, 10, 100)
        muB = self.bohr_magneton
        
        if zeeman_type == 'normal':
            # Upper state (¹P): m = -1, 0, +1
            upper_m1 = -1 * muB * B_range * 1000
            upper_0 = 0 * muB * B_range * 1000
            upper_p1 = 1 * muB * B_range * 1000
            
            # Lower state (¹S): m = 0
            lower_0 = np.zeros_like(B_range)
            
            # Plot upper levels
            self.ax3.plot(B_range, upper_m1, 'cyan', linewidth=2.5, label='Upper: m=-1', linestyle='--')
            self.ax3.plot(B_range, upper_0, 'cyan', linewidth=2.5, label='Upper: m=0')
            self.ax3.plot(B_range, upper_p1, 'cyan', linewidth=2.5, label='Upper: m=+1', linestyle='--')
            
            # Plot lower level
            self.ax3.plot(B_range, lower_0, 'orange', linewidth=2.5, label='Lower: m=0')
            
            # Add reference line at B=0
            self.ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=1)
            
        else:  # Anomalous
            g_upper = 1.5
            g_lower = 1.0
            
            # Upper state (²P): mⱼ = -1, 0, +1
            upper_m1 = -1 * g_upper * muB * B_range * 1000
            upper_0 = 0 * g_upper * muB * B_range * 1000
            upper_p1 = 1 * g_upper * muB * B_range * 1000
            
            # Lower state (²S): mⱼ = -1, 0, +1
            lower_m1 = -1 * g_lower * muB * B_range * 1000
            lower_0 = 0 * g_lower * muB * B_range * 1000
            lower_p1 = 1 * g_lower * muB * B_range * 1000
            
            # Plot upper levels
            self.ax3.plot(B_range, upper_m1, 'cyan', linewidth=2.5, label='Upper: mⱼ=-1', linestyle='--')
            self.ax3.plot(B_range, upper_0, 'cyan', linewidth=2.5, label='Upper: mⱼ=0')
            self.ax3.plot(B_range, upper_p1, 'cyan', linewidth=2.5, label='Upper: mⱼ=+1', linestyle='--')
            
            # Plot lower levels
            self.ax3.plot(B_range, lower_m1, 'orange', linewidth=2.5, label='Lower: mⱼ=-1', linestyle='--')
            self.ax3.plot(B_range, lower_0, 'orange', linewidth=2.5, label='Lower: mⱼ=0')
            self.ax3.plot(B_range, lower_p1, 'orange', linewidth=2.5, label='Lower: mⱼ=+1', linestyle='--')
            
            # Add reference line at B=0
            self.ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=1)
        
        # Mark current field position
        self.ax3.axvline(x=current_B, color='#ff00ff', linestyle=':', linewidth=2.5, 
                        alpha=0.8, label=f'Current B = {current_B:.1f} T')
        
        # Styling
        self.ax3.set_xlabel('Magnetic Field Strength (Tesla)', fontsize=12, 
                           fontweight='bold', color='white')
        self.ax3.set_ylabel('Energy Shift (meV)', fontsize=12, 
                           fontweight='bold', color='white')
        self.ax3.set_title('Energy Level Splitting vs. Magnetic Field Strength', 
                          fontsize=14, fontweight='bold', color='#00aaff')
        
        self.ax3.legend(loc='upper left', fontsize=8, ncol=2,
                       facecolor='#1a1a1a', edgecolor='#00aaff', framealpha=0.9)
        
        self.ax3.grid(True, alpha=0.2, color='#444444')
        self.ax3.spines['bottom'].set_color('#444444')
        self.ax3.spines['top'].set_color('#444444')
        self.ax3.spines['left'].set_color('#444444')
        self.ax3.spines['right'].set_color('#444444')
        self.ax3.tick_params(colors='white')
        
        # Add annotations
        self.ax3.text(0.02, 0.98, f'Zeeman Type: {zeeman_type.capitalize()}', 
                     transform=self.ax3.transAxes, fontsize=10,
                     verticalalignment='top', color='#ffff00',
                     bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.7))
    
    def update_plots(self):
        # Update field label
        B = self.magnetic_field.get()
        self.field_label.config(text=f"{B:.1f} T")
        
        # Update element info
        element = self.get_current_element()
        self.element_info.config(
            text=f"{element['symbol']} - {element['wavelength']} nm • {element['line']}")
        
        # Update plots
        self.plot_energy_levels()
        self.plot_spectrum()
        self.plot_energy_vs_field()
        
        # Apply spacing between plots
        self.fig.subplots_adjust(hspace=0.4, top=0.97, bottom=0.05, left=0.1, right=0.95)
        self.canvas.draw()
    
    def toggle_animation(self):
        """Toggle animation on/off"""
        self.is_running = not self.is_running
        
        if self.is_running:
            self.run_button.config(text="⏸ Pause", bg='#ff6600')
            self.animate()
        else:
            self.run_button.config(text="▶ Run Simulation", bg='#00aa00')
    
    def animate(self):
        """Animate the magnetic field"""
        if not self.is_running:
            return
        
        current_field = self.magnetic_field.get()
        speed = self.animation_speed.get()
        
        # Oscillate the field between 0 and 10 T
        new_field = current_field + (speed * self.animation_direction)
        
        if new_field >= 10:
            new_field = 10
            self.animation_direction = -1
        elif new_field <= 0:
            new_field = 0
            self.animation_direction = 1
        
        self.magnetic_field.set(new_field)
        self.update_plots()
        
        # Schedule next animation frame
        self.root.after(50, self.animate)
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.is_running = False
        self.run_button.config(text="▶ Run Simulation", bg='#00aa00')
        self.magnetic_field.set(0.0)
        self.animation_direction = 1
        self.update_plots()


if __name__ == "__main__":
    root = tk.Tk()
    app = ZeemanSimulator(root)
    root.mainloop()
