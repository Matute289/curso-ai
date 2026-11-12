import asyncio

from llm.circuit_breaker import CircuitBreaker


def test_circuit_breaker():
    """Verifica los estados y la recuperación del Circuit Breaker."""

    async def scenario():
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.2)

        assert await breaker.get_state() == "CLOSED"

        await breaker.record_failure()
        assert await breaker.get_state() == "CLOSED"

        await breaker.record_failure()
        assert await breaker.get_state() == "OPEN"
        assert not await breaker.allow_request()

        await asyncio.sleep(0.3)

        # Pasado el tiempo de recuperación se permite una sola prueba.
        assert await breaker.allow_request()
        assert await breaker.get_state() == "HALF_OPEN"
        assert not await breaker.allow_request()

        await breaker.record_success()
        assert await breaker.get_state() == "CLOSED"

    asyncio.run(scenario())


if __name__ == "__main__":
    test_circuit_breaker()
    print("OK")
