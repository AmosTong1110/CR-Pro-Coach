# Dataset spec
* Image: JPEG, 568x896 RGB, the game arena only (no hand, deck bar, overlays).
* Label: one `.txt` per image, rows `class_id cx cy w h`, normalised to 0..1 (YOLO).
* Classes: ids 0..186 in `classes.txt` order. Ids never change; new classes are appended.
* Splits: `images/train`, `images/val`; whole matches are held out (never single frames).
* `video_labels/<id>/<frame>.txt`: same label format; `frames.csv` gives video id, match, time in seconds
  (video time at 2 fps); `layout.json` gives the crop box (x, y, w, h in source pixels) that maps the
  video frame to the 568x896 arena. Frames contain painted-out objects in the original training set that
  rebuilt frames will not have, so a few real objects can appear unlabelled.
