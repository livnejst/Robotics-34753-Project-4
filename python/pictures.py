"""
Captura d'imatges per a la calibració de la càmera.

  ESPAI -> fa una foto i la desa
  Q/ESC -> surt

La foto que es desa és la imatge NETA (sense els dibuixos de la previsualització).
A la previsualització es veu en verd si detecta el tauler: si no surt en verd,
aquella foto no servirà per calibrar.

Ús:
  python capture_calib.py                 # càmera 0, carpeta calib_images
  python capture_calib.py --cam 1         # una altra càmera (la del robot)
  python capture_calib.py --out fotos --width 1280 --height 720
"""
import argparse
import os
import sys
from datetime import datetime

import cv2

PATTERN = (9, 6)  # cantonades interiors del tauler (7x10 quadrats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="índex de la càmera")
    ap.add_argument("--out", default="calib_images", help="carpeta de sortida")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # CAP_DSHOW obre la càmera molt més ràpid a Windows
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.cam, backend)
    if not cap.isOpened():
        sys.exit(f"No puc obrir la càmera {args.cam}. Prova amb --cam 1, 2...")

    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Càmera {args.cam} oberta a {w}x{h}. ESPAI = foto, Q = sortir.")

    n_saved = len([f for f in os.listdir(args.out) if f.lower().endswith(".png")])

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No arriba imatge de la càmera.")
            break

        preview = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        )
        if found:
            cv2.drawChessboardCorners(preview, PATTERN, corners, found)

        color = (0, 200, 0) if found else (0, 0, 255)
        status = "TAULER OK" if found else "no es veu el tauler"
        cv2.putText(preview, f"{status} | fotos: {n_saved}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Calibracio - ESPAI foto, Q sortir", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            name = datetime.now().strftime("calib_%Y%m%d_%H%M%S_%f")[:-3] + ".png"
            path = os.path.join(args.out, name)
            cv2.imwrite(path, frame)  # imatge neta
            n_saved += 1
            print(f"[{n_saved}] {path}" + ("" if found else "  (ATENCIÓ: tauler no detectat)"))
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Fet. {n_saved} fotos a '{args.out}'.")


if __name__ == "__main__":
    main()