
import asyncio
import time


class CircuitBreaker:
    """
    Controla la disponibilidad de un proveedor mediante el patrón Circuit Breaker.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        """
        Inicializa el circuito con un umbral de fallos y tiempo de recuperación.

        Args:
            failure_threshold: Cantidad de errores consecutivos para abrir el circuito.
            recovery_timeout: Segundos que debe permanecer abierto.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.state = "CLOSED"
        self.opened_at = None

        self._lock = asyncio.Lock()
        self._probe_in_progress = False

    async def allow_request(self) -> bool:
        """
        Determina si el proveedor puede recibir una nueva solicitud.

        Permite una única solicitud de prueba cuando termina el período de
        recuperación del circuito.
        """
        async with self._lock:

            if self.state == "CLOSED":
                return True

            if self.state == "OPEN":
                elapsed = time.monotonic() - self.opened_at

                if elapsed < self.recovery_timeout:
                    return False

                self.state = "HALF_OPEN"
                self._probe_in_progress = False

            if self.state == "HALF_OPEN":
                if self._probe_in_progress:
                    return False

                self._probe_in_progress = True
                return True

            return False

    async def record_success(self):
        """Registra una operación exitosa y restablece el circuito."""
        async with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"
            self.opened_at = None
            self._probe_in_progress = False

    async def record_failure(self):
        """Registra un fallo y abre el circuito cuando se supera el umbral."""
        async with self._lock:

            self.failure_count += 1

            if (
                self.state == "HALF_OPEN"
                or self.failure_count >= self.failure_threshold
            ):
                self.state = "OPEN"
                self.opened_at = time.monotonic()

            self._probe_in_progress = False

    async def release_probe(self):
        """Libera una prueba HALF_OPEN cuando no se pudo completar."""
        async with self._lock:
            if self.state == "HALF_OPEN":
                self._probe_in_progress = False

    async def get_state(self) -> str:
        """Devuelve el estado actual del circuito."""
        async with self._lock:
            return self.state
