import sys
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging

# Importing the DataProcessingModule from another module for clarity and separation of concerns
from data_processing_module import DataProcessingModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MplCanvas(FigureCanvas):
    """A dynamic matplotlib canvas that updates itself with new data."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    def plot(self, data, title):
        """Plot the given data on the canvas."""
        self.axes.clear()
        self.axes.plot(data['time'], data['stress_level'], marker='o', linestyle='-')
        self.axes.set_title(title)
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Stress Level')
        self.axes.tick_params(axis='x', rotation=45)
        self.draw()

class ECGInputForm(QWidget):
    def __init__(self, dataProcessingModule):
        super().__init__()
        self.dpm = dataProcessingModule
        self.setWindowTitle('ECG Based Stress Analysis System')
        self.setGeometry(100, 100, 500, 400)
        self.userSessionData = pd.DataFrame(columns=['timestamp', 'stress_level'])
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Creating input fields for the ECG parameters
        self.hrvInput = QLineEdit()
        self.qrsComplexInput = QLineEdit()
        self.rrIntervalsInput = QLineEdit()
        self.frequencyDomainFeaturesInput = QLineEdit()

        # Adding input fields to the layout with labels
        layout.addLayout(self.createInputFieldLayout('Heart Rate Variability (HRV):', self.hrvInput))
        layout.addLayout(self.createInputFieldLayout('QRS Complex:', self.qrsComplexInput))
        layout.addLayout(self.createInputFieldLayout('R-R Intervals:', self.rrIntervalsInput))
        layout.addLayout(self.createInputFieldLayout('Frequency Domain Features:', self.frequencyDomainFeaturesInput))

        submitBtn = QPushButton('Submit')
        submitBtn.clicked.connect(self.onSubmit)
        layout.addWidget(submitBtn)

        # Canvas for matplotlib plots
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        # Buttons for daily and historical stress levels
        dailyBtn = QPushButton('Show Daily Stress Levels')
        dailyBtn.clicked.connect(lambda: self.showStressLevels('daily'))
        layout.addWidget(dailyBtn)

        historicalBtn = QPushButton('Show Historical Stress Levels')
        historicalBtn.clicked.connect(lambda: self.showStressLevels('historical'))
        layout.addWidget(historicalBtn)

        self.setLayout(layout)

    def createInputFieldLayout(self, labelText, inputWidget):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(labelText))
        layout.addWidget(inputWidget)
        return layout

    def onSubmit(self):
        try:
            inputs = [self.hrvInput.text(), self.qrsComplexInput.text(), self.rrIntervalsInput.text(), self.frequencyDomainFeaturesInput.text()]
            processed_data, valid = self.dpm.validate_input(*inputs)
            if not valid:
                QMessageBox.warning(self, 'Input Error', 'Invalid input data. Please ensure all fields are numeric.')
                return
            
            stress_level = self.dpm.classify_stress_level(processed_data)
            if stress_level is not None:
                QMessageBox.information(self, 'Analysis Complete', f'Estimated Stress Level: {stress_level}')
                new_row = pd.DataFrame({'timestamp': [datetime.now()], 'stress_level': [stress_level]})
                self.userSessionData = pd.concat([self.userSessionData, new_row], ignore_index=True)
                logging.info(f"Data submitted and stress level predicted: {stress_level}")
            else:
                QMessageBox.warning(self, 'Prediction Error', 'Error during stress level prediction. Please try again.')
                logging.error("Error during stress level prediction: No stress level returned.")
        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Invalid input data. Please ensure all fields are numeric.')
            logging.error("Invalid input - Non-numeric values detected.")
        except Exception as e:
            QMessageBox.warning(self, 'Prediction Error', 'Error during stress level prediction.')
            logging.error("Error during stress level prediction: ", exc_info=True)

    def showStressLevels(self, mode):
        if self.userSessionData.empty:
            QMessageBox.warning(self, 'No Data', 'No stress level data available.')
            return
        data = self.userSessionData.copy()
        data['time'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if mode == 'daily':
            today = datetime.now().date()
            data = data[data['timestamp'].dt.date == today]
            if data.empty:
                QMessageBox.warning(self, 'No Data', 'No stress level data available for today.')
                return
        self.canvas.plot(data, f"{mode.capitalize()} Stress Levels")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dpm = DataProcessingModule()  # Instantiate DataProcessingModule without passing dataset path argument
    ex = ECGInputForm(dpm)
    ex.show()
    sys.exit(app.exec_())