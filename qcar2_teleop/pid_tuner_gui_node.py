#!/usr/bin/env python3
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import rclpy
from rclpy.node import Node

from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.msg import ParameterType


class PIDTunerGUI(Node):
    """
    GUI para tunear kp/ki/kd (y opcional Kalman) en tiempo real.
    - Sliders -> llama al servicio /<target_node>/set_parameters
    - Botón Guardar -> crea YAML con valores actuales
    - Botón Leer -> llama /<target_node>/get_parameters
    """

    def __init__(self):
        super().__init__('pid_tuner_gui')

        # --- Config ROS ---
        self.declare_parameter('target_node', 'yellow_line_follower_controller')
        self.declare_parameter('push_rate_hz', 10.0)
        self.declare_parameter('enable_kalman_tuning', True)

        self.target_node = self.get_parameter('target_node').value
        self.push_rate_hz = float(self.get_parameter('push_rate_hz').value)
        self.enable_kalman_tuning = bool(self.get_parameter('enable_kalman_tuning').value)

        # Servicios estándar de parámetros en ROS2
        self.get_srv_name = f'/{self.target_node}/get_parameters'
        self.set_srv_name = f'/{self.target_node}/set_parameters'

        self.cli_get = self.create_client(GetParameters, self.get_srv_name)
        self.cli_set = self.create_client(SetParameters, self.set_srv_name)

        # --- Estado ---
        self._last_sent = {'kp': None, 'ki': None, 'kd': None, 'use_kalman': None, 'kalman_q': None, 'kalman_r': None}
        self._dirty = False
        self._stop = False

        # =========================
        # ✅ IMPORTANTE: Primero Tk()
        # =========================
        self.root = tk.Tk()
        self.root.title("PID Tuner (ROS2) - QCar2")
        self.root.geometry("540x450")

        # ✅ Variables Tk DESPUÉS del root (y con master)
        self.gui_kp = tk.DoubleVar(master=self.root, value=1.2)
        self.gui_ki = tk.DoubleVar(master=self.root, value=0.0)
        self.gui_kd = tk.DoubleVar(master=self.root, value=0.08)
        self.gui_use_kalman = tk.BooleanVar(master=self.root, value=True)
        self.gui_q = tk.DoubleVar(master=self.root, value=0.02)
        self.gui_r = tk.DoubleVar(master=self.root, value=0.08)

        # Construir GUI
        self._build_gui()

        # Thread que empuja cambios
        self._push_thread = threading.Thread(target=self._push_loop, daemon=True)
        self._push_thread.start()

        # Leer valores iniciales del controlador
        self._try_load_from_controller_async()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- GUI ----------------
    def _build_gui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frm, text=f"Target node: {self.target_node}", font=("Arial", 12, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        self._add_slider(frm, "Kp", self.gui_kp, 0.0, 5.0)
        self._add_slider(frm, "Ki", self.gui_ki, 0.0, 2.0)
        self._add_slider(frm, "Kd", self.gui_kd, 0.0, 2.0)

        ttk.Separator(frm).pack(fill=tk.X, pady=10)

        chk = ttk.Checkbutton(frm, text="Usar Kalman", variable=self.gui_use_kalman, command=self._mark_dirty)
        chk.pack(anchor="w")

        if self.enable_kalman_tuning:
            self._add_slider(frm, "Kalman q (process noise)", self.gui_q, 0.0001, 0.2)
            self._add_slider(frm, "Kalman r (measurement noise)", self.gui_r, 0.0001, 0.5)

        ttk.Separator(frm).pack(fill=tk.X, pady=10)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X)

        ttk.Button(btns, text="Aplicar ahora", command=self._send_once).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Guardar valores (YAML)", command=self._save_yaml).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Leer del controlador", command=self._try_load_from_controller_async).pack(side=tk.LEFT)

        self.status = ttk.Label(frm, text="Listo", foreground="gray")
        self.status.pack(anchor="w", pady=(12, 0))

    def _add_slider(self, parent, label, var, lo, hi):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=4)

        ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)

        s = ttk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        e = ttk.Entry(row, width=10)
        e.pack(side=tk.RIGHT)
        e.insert(0, f"{var.get():.6f}")

        def on_move(v):
            try:
                var.set(float(v))
            except Exception:
                pass
            self._mark_dirty()

        s.configure(command=on_move)
        s.set(var.get())

        def refresh_entry(*_):
            e.delete(0, tk.END)
            e.insert(0, f"{var.get():.6f}")

        var.trace_add("write", refresh_entry)

    def _mark_dirty(self):
        self._dirty = True

    def run(self):
        self.root.mainloop()

    def _on_close(self):
        self._stop = True
        self.root.destroy()

    # ---------------- ROS parameter services helpers ----------------
    def _wait_services(self, timeout_sec=2.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            ok_get = self.cli_get.wait_for_service(timeout_sec=0.2)
            ok_set = self.cli_set.wait_for_service(timeout_sec=0.2)
            if ok_get and ok_set:
                return True
        return False

    def _make_param(self, name: str, value):
        p = Parameter()
        p.name = name
        pv = ParameterValue()

        if isinstance(value, bool):
            pv.type = ParameterType.PARAMETER_BOOL
            pv.bool_value = bool(value)
        else:
            pv.type = ParameterType.PARAMETER_DOUBLE
            pv.double_value = float(value)

        p.value = pv
        return p

    def _set_params(self, desired: dict) -> bool:
        if not self._wait_services(timeout_sec=2.0):
            self.status.config(text="No veo servicios de parámetros (¿controlador corriendo?)", foreground="red")
            return False

        req = SetParameters.Request()
        req.parameters = [
            self._make_param('kp', desired['kp']),
            self._make_param('ki', desired['ki']),
            self._make_param('kd', desired['kd']),
            self._make_param('use_kalman', desired['use_kalman']),
            self._make_param('kalman_q', desired['kalman_q']),
            self._make_param('kalman_r', desired['kalman_r']),
        ]

        fut = self.cli_set.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)

        try:
            res = fut.result()
            ok = all([r.successful for r in res.results])
            return ok
        except Exception:
            return False

    def _get_params(self, names) -> dict:
        if not self._wait_services(timeout_sec=2.0):
            return {}

        req = GetParameters.Request()
        req.names = list(names)

        fut = self.cli_get.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)

        out = {}
        try:
            res = fut.result()
            for name, val in zip(names, res.values):
                if val.type == ParameterType.PARAMETER_BOOL:
                    out[name] = val.bool_value
                else:
                    out[name] = val.double_value
        except Exception:
            pass
        return out

    # ---------------- GUI actions ----------------
    def _try_load_from_controller_async(self):
        def worker():
            self.status.config(text="Leyendo del controlador...", foreground="orange")
            names = ['kp', 'ki', 'kd', 'use_kalman', 'kalman_q', 'kalman_r']
            vals = self._get_params(names)
            if not vals:
                self.status.config(text="No pude leer (¿controlador corriendo? ¿nombre correcto?)", foreground="red")
                return

            self.gui_kp.set(float(vals.get('kp', self.gui_kp.get())))
            self.gui_ki.set(float(vals.get('ki', self.gui_ki.get())))
            self.gui_kd.set(float(vals.get('kd', self.gui_kd.get())))
            self.gui_use_kalman.set(bool(vals.get('use_kalman', self.gui_use_kalman.get())))
            self.gui_q.set(float(vals.get('kalman_q', self.gui_q.get())))
            self.gui_r.set(float(vals.get('kalman_r', self.gui_r.get())))

            self._dirty = False
            self.status.config(text="Parámetros cargados del controlador", foreground="green")

        threading.Thread(target=worker, daemon=True).start()

    def _send_once(self):
        self._send_params(force=True)

    def _send_params(self, force=False):
        kp = float(self.gui_kp.get())
        ki = float(self.gui_ki.get())
        kd = float(self.gui_kd.get())
        use_k = bool(self.gui_use_kalman.get())
        q = float(self.gui_q.get())
        r = float(self.gui_r.get())

        desired = {'kp': kp, 'ki': ki, 'kd': kd, 'use_kalman': use_k, 'kalman_q': q, 'kalman_r': r}

        changed = force
        if not changed:
            for k, v in desired.items():
                if self._last_sent.get(k) != v:
                    changed = True
                    break
        if not changed:
            return

        ok = self._set_params(desired)
        if ok:
            self._last_sent.update(desired)
            self._dirty = False
            self.status.config(text=f"Aplicado: kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}", foreground="green")
        else:
            self.status.config(text="Falló set_parameters (¿patch en controlador?)", foreground="red")

    def _push_loop(self):
        period = 1.0 / max(1.0, self.push_rate_hz)
        while not self._stop:
            time.sleep(period)
            if self._dirty:
                self._send_params(force=False)

    def _save_yaml(self):
        kp = float(self.gui_kp.get())
        ki = float(self.gui_ki.get())
        kd = float(self.gui_kd.get())
        use_k = bool(self.gui_use_kalman.get())
        q = float(self.gui_q.get())
        r = float(self.gui_r.get())

        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            initialfile="pid_tuned.yaml",
            filetypes=[("YAML", "*.yaml"), ("All files", "*.*")]
        )
        if not path:
            return

        yaml_text = (
            "yellow_line_follower_controller:\n"
            "  ros__parameters:\n"
            f"    kp: {kp}\n"
            f"    ki: {ki}\n"
            f"    kd: {kd}\n"
            f"    use_kalman: {str(use_k).lower()}\n"
            f"    kalman_q: {q}\n"
            f"    kalman_r: {r}\n"
        )

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_text)
            self.status.config(text=f"Guardado en: {path}", foreground="blue")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main(args=None):
    rclpy.init(args=args)
    node = PIDTunerGUI()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
