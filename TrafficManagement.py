import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SmartTrafficManager:
    def __init__(self, num_lanes=4, history_length=10, cycle_length=120):
        self.num_lanes = num_lanes
        self.history_length = history_length
        self.cycle_length = cycle_length
        self.vehicle_counts = np.zeros((history_length, num_lanes))
        self.signal_timings = np.full(num_lanes, cycle_length / num_lanes)
    
    def update_vehicle_counts(self, new_counts):
        self.vehicle_counts[:-1] = self.vehicle_counts[1:]
        self.vehicle_counts[-1] = new_counts

    def predict_congestion(self):
        predicted_counts = np.zeros(self.num_lanes)
        X = np.arange(self.history_length)
        X_design = np.vstack([X, np.ones(self.history_length)]).T
        
        for lane in range(self.num_lanes):
            y = self.vehicle_counts[:, lane]
            beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            x_next = np.array([self.history_length, 1])
            pred = beta.dot(x_next)
            predicted_counts[lane] = max(pred, 0)
        
        return predicted_counts
    
    def optimize_signal_timings(self, predicted_counts):
        total = predicted_counts.sum()
        if total == 0:
            self.signal_timings = np.full(self.num_lanes, self.cycle_length / self.num_lanes)
        else:
            self.signal_timings = (predicted_counts / total) * self.cycle_length
    
    def simulate_step(self, new_counts):
        self.update_vehicle_counts(new_counts)
        predicted = self.predict_congestion()
        self.optimize_signal_timings(predicted)
        return {
            'latest_counts': new_counts,
            'predicted_counts': predicted,
            'signal_timings': self.signal_timings
        }

# ------------------------ GUI Implementation ------------------------

class TrafficGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Traffic Manager")
        self.root.configure(bg='#e0f7fa')

        self.manager = None
        self.time_step = 0
        self.latest_counts_list = []
        self.predicted_counts_list = []
        self.signal_timings_list = []
        self.time_steps = []

        self.build_gui()

    def build_gui(self):
        self.title_label = tk.Label(self.root, text="Smart Traffic Management System", font=("Helvetica", 18, "bold"), bg='#80deea', fg='black', pady=10)
        self.title_label.pack(fill=tk.X)

        frame = tk.Frame(self.root, bg='#e0f7fa')
        frame.pack(pady=10)

        tk.Label(frame, text="Enter number of lanes:", bg='#e0f7fa').grid(row=0, column=0, padx=5, pady=5)
        self.lanes_entry = tk.Entry(frame)
        self.lanes_entry.grid(row=0, column=1, padx=5, pady=5)

        self.start_button = tk.Button(frame, text="Start Simulation", bg="#00796b", fg="white", command=self.start_simulation)
        self.start_button.grid(row=0, column=2, padx=5)

        self.execute_button = tk.Button(frame, text="Execute Steps", bg="#388e3c", fg="white", command=self.execute_steps, state=tk.DISABLED)
        self.execute_button.grid(row=0, column=3, padx=5)

        self.plot_button = tk.Button(frame, text="Show Graphs", bg="#5d4037", fg="white", command=self.show_graphs, state=tk.DISABLED)
        self.plot_button.grid(row=0, column=4, padx=5)

        self.output_text = tk.Text(self.root, height=10, width=100, bg='#ffffff', fg='black')
        self.output_text.pack(pady=10)

    def start_simulation(self):
        try:
            num_lanes = int(self.lanes_entry.get())
            if num_lanes <= 0:
                raise ValueError
            self.manager = SmartTrafficManager(num_lanes=num_lanes)
            self.num_lanes = num_lanes
            self.base_counts = np.random.randint(10, 25, size=num_lanes)
            self.output_text.insert(tk.END, f"Simulation started for {num_lanes} lanes.\n")
            self.execute_button.config(state=tk.NORMAL)
            self.plot_button.config(state=tk.DISABLED)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a positive integer for number of lanes.")

    def execute_steps(self):
        if self.time_step < self.num_lanes:
            self.simulate_step()
            self.root.after(2000, self.execute_steps)  # Delay of 2 seconds between steps
        else:
            self.plot_button.config(state=tk.NORMAL)

    def simulate_step(self):
        if self.manager is None:
            return
        
        noise = np.random.randint(-5, 6, size=self.num_lanes)
        trend = (self.time_step // 5) * 2
        new_counts = self.base_counts + noise + trend
        new_counts = np.clip(new_counts, 0, None)

        results = self.manager.simulate_step(new_counts)

        self.time_step += 1
        self.time_steps.append(self.time_step)
        self.latest_counts_list.append(results['latest_counts'])
        self.predicted_counts_list.append(results['predicted_counts'])
        self.signal_timings_list.append(results['signal_timings'])

        self.output_text.insert(tk.END, f"\nTime Step {self.time_step}:\n")
        self.output_text.insert(tk.END, f"  Latest Counts: {results['latest_counts']}\n")
        self.output_text.insert(tk.END, f"  Predicted Counts: {results['predicted_counts'].round(1)}\n")
        self.output_text.insert(tk.END, f"  Signal Timings (s): {results['signal_timings'].round(1)}\n")
        
        # Auto-scroll to the latest step
        self.output_text.see(tk.END)

    def show_graphs(self):
        # Clear any existing graph frame
        if hasattr(self, 'graph_frame'):
            self.graph_frame.destroy()

        # Create a new frame for the graphs
        self.graph_frame = tk.Frame(self.root, bg='#e0f7fa')
        self.graph_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Prepare the data for plotting
        latest_counts_array = np.array(self.latest_counts_list)
        predicted_counts_array = np.array(self.predicted_counts_list)
        signal_timings_array = np.array(self.signal_timings_list)

        # Create a matplotlib figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.tight_layout(pad=3.0)

        # Plot latest vehicle counts
        for lane in range(self.num_lanes):
            axes[0].plot(self.time_steps, latest_counts_array[:, lane], label=f'Lane {lane + 1}')
        axes[0].set_title('Latest Vehicle Counts Over Time')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Vehicle Count')
        axes[0].legend()
        axes[0].grid(True)

        # Plot predicted vehicle counts
        for lane in range(self.num_lanes):
            axes[1].plot(self.time_steps, predicted_counts_array[:, lane], label=f'Lane {lane + 1}')
        axes[1].set_title('Predicted Vehicle Counts Over Time')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Predicted Count')
        axes[1].legend()
        axes[1].grid(True)

        # Plot signal timings
        for lane in range(self.num_lanes):
            axes[2].plot(self.time_steps, signal_timings_array[:, lane], label=f'Lane {lane + 1}')
        axes[2].set_title('Optimized Signal Timings Over Time')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Signal Timing (s)')
        axes[2].legend()
        axes[2].grid(True)

        # Embed the matplotlib figure into the Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

# ------------------------ Run GUI ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    gui = TrafficGUI(root)
    root.mainloop()