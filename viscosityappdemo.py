import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class ViscosityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Viscosity App Demo")
        self.setGeometry(100, 100, 1024, 768)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        video_button = QPushButton("Select Video Location")
        video_button.clicked.connect(self.select_video_location)
        layout.addWidget(video_button)

        viscosity_button = QPushButton("Calculate Viscosity")
        viscosity_button.clicked.connect(self.calculate_viscosity)
        layout.addWidget(viscosity_button)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        widget = QWidget()
        widget.setLayout(layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(widget)

        self.setLayout(main_layout)
        self.show()

        self.viscosity_data = []

    def select_video_location(self):
        video_location, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if video_location:
            self.video_location = video_location

    def create_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Viscosity (poise)")

        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        ax.set_yticks(np.arange(0.002, 0.030, 0.002))

        valid_viscosity_data = [v for v in self.viscosity_data if not np.isinf(v)]
        ax.set_ylim([0, np.max(valid_viscosity_data) + 0.02])

        ax.tick_params(axis='y', labelsize=12)
        ax.yaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(length=8, direction='in')

        plt.ylim([0, 1])
        plt.yticks(np.linspace(0, 1, 33))

        time = np.arange(len(self.viscosity_data))
        ax.plot(time, self.viscosity_data, 'b-')

        ax.yaxis.set_label_coords(-0.08, 0.5)

        self.figure.set_size_inches(10, 6)
        self.figure.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95)

        self.canvas.draw()

    def calculate_viscosity(self):
        if not hasattr(self, 'video_location'):
            QMessageBox.warning(self, "Error", "Please select a video location first.")
            return

        cap = cv2.VideoCapture(self.video_location)
        fps = cap.get(cv2.CAP_PROP_FPS)

        lower_purple = np.array([125, 50, 50])
        upper_purple = np.array([175, 255, 255])

        displacement_data = []
        frame_count = 0
        frame_interval = int(fps)
        draw_start_time = time.time() + 7

        cv2.namedWindow('Viscosity')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_purple, upper_purple)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0 and frame_count > 10 * fps:
                max_contour = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
                x = int(x)
                y = int(y)
                radius = int(radius)

                if len(displacement_data) > 0:
                    last_position = displacement_data[-1]
                    distance = np.sqrt((x - last_position[0]) ** 2 + (y - last_position[1]) ** 2)
                    displacement_data.append([x, y])
                else:
                    displacement_data.append([x, y])

                if len(displacement_data) >= 2:
                    displacements = np.array(displacement_data)
                    displacements = np.sqrt(np.sum((displacements - displacements[0]) ** 2, axis=1))
                    acf_data = np.correlate(displacements, displacements, mode='full')
                    acf_data = acf_data[len(acf_data) // 2:]
                    acf_data /= np.max(acf_data)
                    tau = np.arange(len(acf_data)) / fps
                    popt, pcov = self.fit_SED(tau, acf_data)
                    viscosity = self.calculate_and_append_viscosity(radius, fps, displacements)
                    if np.isnan(viscosity):
                        QMessageBox.warning(self, "Error", "Viscosity calculation failed.")
                    else:
                        print(f"Current calculated viscosity: {viscosity:.3f} poise")

                    cv2.putText(frame, f"Viscosity: {viscosity:.3f} poise", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.drawContours(frame, [max_contour], 0, (0, 255, 0), 2)
                    
                    cv2.imshow('Viscosity', frame)

                    if frame_count % frame_interval == 0 and time.time() >= draw_start_time:
                        self.viscosity_data.append(viscosity)
                        self.create_plot()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    @staticmethod
    def calculate_and_append_viscosity(radius, fps, displacement_data):
        T = 25 + 273
        D = np.mean(displacement_data ** 2) / (6 * fps)
        try:
            eta = T / (6 * np.pi * D * radius) * 10
        except ZeroDivisionError:
            eta = np.nan
        return eta

    @staticmethod
    def fit_SED(tau, g):
        popt, pcov = curve_fit(ViscosityApp.SED, tau, g, p0=[g[0], 1e-6], maxfev=10000)
        return popt, pcov

    @staticmethod
    def SED(x, A, D):
        return A / (1 + x ** 2 * D)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viscosity_app = ViscosityApp()
    sys.exit(app.exec_())
