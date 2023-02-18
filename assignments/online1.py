from shared.Calibrator import Calibrator
from shared.LiveCamera import LiveCamera

# Without manual files (does not matter if not recalibrating)
calibrator = Calibrator('resources/img_*[!X].png')

calibrator.calibrate(recalibrate=False, save=False, savename='calibration.npz')

live = LiveCamera(calibrator)
live.start()


# Miscellaneous lines for testing
# calibrator = Calibrator('resources/img_26_X.png')
# calibrator = Calibrator('resources/img_0.png')
# calibrator = Calibrator('resources/img_*.png')
# calibrator.calibrate(recalibrate=True)

