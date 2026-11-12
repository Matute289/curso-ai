
class LLMError(Exception):
    """Error base para las operaciones del administrador LLM."""

    def __init__(self, message: str, provider: str = None):
        """Inicializa un error con su mensaje y proveedor asociado."""
        super().__init__(message)
        self.provider = provider


class LLMConnectionError(LLMError):
    """Representa errores de conexión o disponibilidad del proveedor."""


class LLMRateLimitError(LLMError):
    """Representa errores por exceso de solicitudes o rate limiting."""


class LLMProviderError(LLMError):
    """Representa errores generales devueltos por el proveedor."""


class LLMStreamInterruptedError(LLMError):
    """Indica que el streaming se interrumpió después de enviar texto."""
