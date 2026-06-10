# 🎓 Face Attendance — Reconocimiento Facial Académico

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/DeepFace-ArcFace-00B4D8?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/uv-package_manager-7C3AED?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

<p align="center">
  Prueba de concepto de un sistema de reconocimiento facial para identificación de personas,<br/>
  construido con DeepFace (ArcFace) + RetinaFace + Streamlit. Sin entrenamiento de modelos.
</p>

---

## ¿Qué hace?

| Página | Funcionalidad |
|--------|--------------|
| **Estudiantes** | Registra personas con nombre y fotos de referencia. Muestra en tiempo real si cada foto tiene un rostro detectable antes de guardar. Permite agregar fotos extra a perfiles ya existentes. |
| **Identificar** | Sube una foto o usa la cámara. El sistema compara el rostro contra todos los perfiles registrados y muestra el nombre y porcentaje de confianza. |

## Características técnicas

- **Sin entrenamiento** — usa embeddings preentrenados de ArcFace (512 dimensiones).
- **RetinaFace como detector** — red neuronal mucho más precisa que los Haar Cascades clásicos.
- **Bounding boxes en tiempo real** — verde para el rostro principal, amarillo para secundarios, rojo si no se detecta nada.
- **Embeddings incrementales** — agregar fotos al perfil de una persona promedia los nuevos embeddings con el existente.
- **Base de datos local** — SQLite con SQLAlchemy 2.x. Sin infraestructura externa.
- **Gestión de paquetes con `uv`** — instalación reproducible y rápida.

---

## Cómo funciona

```
Foto de entrada
      │
      ▼
RetinaFace (detector)
      │  detecta región facial
      ▼
ArcFace (embedding)
      │  vector de 512 dimensiones
      ▼
Similitud coseno contra galería
      │  compara contra todos los perfiles
      ▼
Mejor match ≥ umbral → identidad
```

El threshold por defecto es `0.40` (distancia coseno). Se puede ajustar en `.env`.

---

## Requisitos previos

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — gestor de paquetes
- Cámara web (opcional, para captura en vivo)
- ~2 GB de espacio para los modelos de DeepFace (se descargan automáticamente la primera vez)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/AndresInsuasty/face-attendance.git
cd face-attendance

# 2. Instalar dependencias
uv sync

# 3. Configurar variables de entorno (opcional)
cp .env.example .env
# Editar .env si quieres cambiar el umbral o el modelo
```

> **Primera ejecución:** DeepFace descarga los pesos de RetinaFace y ArcFace (~500 MB).
> Solo ocurre una vez; quedan en `~/.deepface/weights/`.

---

## Ejecutar la app

```bash
uv run streamlit run src/app.py
```

Abre `http://localhost:8501` en el navegador.

---

## Flujo de uso

### 1. Registrar una persona

1. Ve a **Estudiantes → Nuevo estudiante**
2. Escribe el nombre
3. Sube 3 o más fotos (distintos ángulos, iluminación variada)
4. Verifica que todas tengan bounding box verde
5. Clic en **Registrar estudiante**

### 2. Identificar

1. Ve a **Identificar**
2. Sube una foto o activa la cámara
3. Clic en **Identificar**
4. El sistema muestra el nombre y el porcentaje de confianza

### 3. Mejorar un perfil existente

1. Ve a **Estudiantes → Agregar fotos**
2. Selecciona la persona
3. Sube fotos adicionales
4. Clic en **Actualizar embedding**

---

## Estructura del proyecto

```
face-attendance/
├── src/
│   ├── app.py                   # Entrypoint Streamlit
│   ├── config.py                # Configuración centralizada (pydantic-settings)
│   ├── database/
│   │   ├── models.py            # Modelos SQLAlchemy 2.x
│   │   └── connection.py        # Engine, sesión, init_db
│   ├── repositories/
│   │   └── student_repo.py      # CRUD estudiantes
│   ├── services/
│   │   └── face_service.py      # Embeddings, matching, merge
│   ├── schemas/
│   │   └── student.py           # Validación Pydantic v2
│   └── pages/
│       ├── 1_Estudiantes.py     # Registro + fotos + lista
│       └── 2_Identificar.py     # Identificación en tiempo real
├── tests/
│   ├── conftest.py
│   ├── test_face_service.py
│   └── test_student_repo.py
├── data/
│   ├── db/                      # SQLite (ignorado por git)
│   └── faces/                   # Fotos de referencia (ignoradas por git)
├── pyproject.toml
└── .env.example
```

---

## Configuración

Copia `.env.example` a `.env` y ajusta según necesites:

```env
DEBUG=false

# Modelo de embedding (ArcFace recomendado)
DEEPFACE_MODEL=ArcFace

# Detector de rostros (retinaface >> opencv en precisión)
DEEPFACE_DETECTOR=retinaface

# Umbral de similitud coseno (0.0–1.0)
# Más alto = más estricto. Ajusta según tu caso de uso.
SIMILARITY_THRESHOLD=0.40

# Fotos mínimas requeridas para registrar una persona
ENROLLMENT_PHOTOS_MIN=3
```

---

## Tests

```bash
# Instalar dependencias de desarrollo
uv sync --extra dev

# Ejecutar tests
uv run pytest

# Con reporte de cobertura en HTML
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Calidad de código

```bash
# Linter y formatter
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/
```

---

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Interfaz | [Streamlit](https://streamlit.io/) |
| Reconocimiento facial | [DeepFace](https://github.com/serengil/deepface) · ArcFace + RetinaFace |
| Base de datos | SQLite + [SQLAlchemy 2.x](https://www.sqlalchemy.org/) |
| Validación | [Pydantic v2](https://docs.pydantic.dev/) |
| Gestión de paquetes | [uv](https://docs.astral.sh/uv/) |
| Tests | pytest + pytest-cov |
| Linting | ruff + mypy |

---

## Limitaciones conocidas

- **Velocidad:** RetinaFace es más lento que detectores más simples. El primer enrollment puede tardar algunos segundos mientras carga los modelos.
- **Fotos de grupo:** Si la imagen tiene múltiples rostros, se toma automáticamente el de mayor área.
- **Iluminación:** Como cualquier sistema de reconocimiento facial, el rendimiento baja con iluminación muy pobre o contraluz extremo.
- **Privacidad:** Este sistema almacena embeddings faciales. Para uso en producción, asegúrate de cumplir con la regulación de datos biométricos aplicable (GDPR, Ley 1581 en Colombia, etc.).

---

## Contribuir

1. Haz fork del repositorio
2. Crea una rama: `git checkout -b feature/mi-mejora`
3. Commitea tus cambios: `git commit -m "feat: descripción"`
4. Abre un Pull Request

---

## Licencia

MIT — ver [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  Hecho con ❤️ como prueba de concepto de reconocimiento facial académico
</p>
