"""Ejecuta todo el flujo de análisis en orden (Windows, macOS o Linux).

Uso:
    python run_all.py

Lee los datos de data/raw/, escribe tablas, figuras y el registro en results/
y al final ejecuta todas las pruebas de la carpeta tests/ con pytest.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"

PASOS = [
    "01_reproducir_idr.py",
    "02_replicar_clasificacion.py",
    "03_regresion_multiple.py",
    "04_series_tiempo.py",
    "05_validacion_cruzada.py",
    "06_figuras_y_resumen.py",
]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    log_path = RESULTS / "log_ejecucion.txt"
    inicio = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for paso in PASOS:
            print(f"\n=== {paso} ===", flush=True)
            log.write(f"\n=== {paso} ===\n")
            proc = subprocess.run(
                [sys.executable, paso],
                cwd=SRC,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            print(proc.stdout, end="")
            log.write(proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                log.write(proc.stderr)
                sys.exit(f"Falló {paso}. Revisa {log_path}.")
        print("\n=== Pruebas (tests/) ===", flush=True)
        log.write("\n=== Pruebas (tests/) ===\n")
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v", "-p", "no:cacheprovider"],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(proc.stdout, end="")
        log.write(proc.stdout)
        if proc.returncode != 0:
            sys.exit(f"Alguna prueba falló. Revisa {log_path}.")
    print(f"\nListo en {time.time() - inicio:.0f} s. Resultados en {RESULTS}")


if __name__ == "__main__":
    main()
