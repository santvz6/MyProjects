// --- Configuración ---
const CANVAS_SIZE = 600;
const CELULA_SIZE = 15; // Tamaño de cada célula en píxeles
const COLS = CANVAS_SIZE / CELULA_SIZE;
const ROWS = CANVAS_SIZE / CELULA_SIZE;

// --- Variables del Juego ---
const canvas = document.getElementById('juegoCanvas');
const ctx = canvas.getContext('2d');
let grid; // La cuadrícula principal de células
let running = false;
let intervalId;

// Referencias a los botones y velocidad
const iniciarDetenerBtn = document.getElementById('iniciar-detener');
const reiniciarBtn = document.getElementById('reiniciar');
const velocidadInput = document.getElementById('velocidad');

// Inicializa la cuadrícula con un estado aleatorio
function inicializarCuadricula() {
    grid = new Array(ROWS);
    for (let i = 0; i < ROWS; i++) {
        grid[i] = new Array(COLS);
        for (let j = 0; j < COLS; j++) {
            // Inicializa aleatoriamente: 0 (muerta) o 1 (viva)
            grid[i][j] = Math.random() > 0.7 ? 1 : 0;
        }
    }
}

// Dibuja la cuadrícula en el canvas
function dibujarCuadricula() {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE); // Limpiar
    for (let i = 0; i < ROWS; i++) {
        for (let j = 0; j < COLS; j++) {
            const x = j * CELULA_SIZE;
            const y = i * CELULA_SIZE;

            if (grid[i][j] === 1) {
                // Célula viva (negra)
                ctx.fillStyle = '#000000';
                ctx.fillRect(x, y, CELULA_SIZE, CELULA_SIZE);
            } else {
                // Célula muerta (borde gris para visualización)
                ctx.strokeStyle = '#eeeeee';
                ctx.strokeRect(x, y, CELULA_SIZE, CELULA_SIZE);
            }
        }
    }
}

// Calcula el estado de la siguiente generación
function siguienteGeneracion() {
    // Crear una nueva cuadrícula para almacenar el estado futuro
    let nextGrid = new Array(ROWS);
    for (let i = 0; i < ROWS; i++) {
        nextGrid[i] = new Array(COLS);
    }

    for (let i = 0; i < ROWS; i++) {
        for (let j = 0; j < COLS; j++) {
            const estadoActual = grid[i][j];
            const vecinosVivos = contarVecinosVivos(i, j);

            // --- Reglas del Juego de la Vida de Conway ---

            if (estadoActual === 1) {
                // 1. Superpoblación (muere si tiene más de 3 vecinos vivos)
                // 2. Soledad (muere si tiene menos de 2 vecinos vivos)
                if (vecinosVivos < 2 || vecinosVivos > 3) {
                    nextGrid[i][j] = 0; // Muere
                } else {
                    nextGrid[i][j] = 1; // Vive/Permanece
                }
            } else {
                // 3. Reproducción (cobra vida si tiene exactamente 3 vecinos vivos)
                if (vecinosVivos === 3) {
                    nextGrid[i][j] = 1; // Cobra vida
                } else {
                    nextGrid[i][j] = 0; // Sigue muerta
                }
            }
        }
    }

    grid = nextGrid; // Actualizar la cuadrícula principal
}

// Cuenta los vecinos vivos de una célula (i, j)
function contarVecinosVivos(i, j) {
    let count = 0;
    // Bucle sobre las 8 direcciones posibles (incluyendo las diagonales)
    for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
        for (let colOffset = -1; colOffset <= 1; colOffset++) {
            // No contar la propia célula
            if (rowOffset === 0 && colOffset === 0) {
                continue;
            }

            const newRow = i + rowOffset;
            const newCol = j + colOffset;

            // Verificar límites (toroidal o bordes fijos)
            // Aquí usamos bordes fijos (las células fuera del límite no existen)
            if (newRow >= 0 && newRow < ROWS && newCol >= 0 && newCol < COLS) {
                count += grid[newRow][newCol];
            }
        }
    }
    return count;
}

// Bucle principal del juego
function gameLoop() {
    siguienteGeneracion();
    dibujarCuadricula();
}

// --- Control de Ejecución ---

function iniciarJuego() {
    if (running) return;

    running = true;
    iniciarDetenerBtn.textContent = 'Detener';
    const velocidad = parseInt(velocidadInput.value) || 100;
    intervalId = setInterval(gameLoop, velocidad);
}

function detenerJuego() {
    if (!running) return;

    running = false;
    iniciarDetenerBtn.textContent = 'Iniciar';
    clearInterval(intervalId);
}

function manejarInicioDetencion() {
    if (running) {
        detenerJuego();
    } else {
        iniciarJuego();
    }
}

function reiniciarJuego() {
    detenerJuego();
    inicializarCuadricula();
    dibujarCuadricula();
}

// --- Event Listeners ---

iniciarDetenerBtn.addEventListener('click', manejarInicioDetencion);
reiniciarBtn.addEventListener('click', reiniciarJuego);

// Permite cambiar la velocidad en tiempo real si el juego está corriendo
velocidadInput.addEventListener('change', () => {
    if (running) {
        detenerJuego();
        iniciarJuego(); // Reinicia el intervalo con la nueva velocidad
    }
});

// --- Inicialización ---
inicializarCuadricula();
dibujarCuadricula();

//