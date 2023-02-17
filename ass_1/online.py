from Calibrator import Calibrator
from LiveCamera import LiveCamera

# Without manual files (does not matter if not recalibrating)
calibrator = Calibrator('new_img/img_*[!X].png')

calibrator.calibrate(recalibrate=False, save=False, savename='calibration.npz')

live = LiveCamera(calibrator)
live.start()


# Miscellaneous lines for testing
# calibrator = Calibrator('new_img/img_26_X.png')
# calibrator = Calibrator('new_img/img_0.png')
# calibrator = Calibrator('new_img/img_*.png')
# calibrator = Calibrator('resources/img_20_X.png')
# calibrator.calibrate(recalibrate=True)

