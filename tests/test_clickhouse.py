import httpx
import pytest

from explained_ml.clickhouse import ClickHouseClient, ClickHouseError


def client_over(handler) -> ClickHouseClient:
    transport = httpx.MockTransport(handler)
    return ClickHouseClient(
        "http://clickhouse:8123",
        database="explained",
        client=httpx.AsyncClient(transport=transport, base_url="http://clickhouse:8123"),
    )


async def test_rows_are_parsed_from_json_each_row():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='{"article_id":"a1","score":2.5}\n{"article_id":"a2","score":1}\n')

    rows = await client_over(handler).query_rows("SELECT article_id, score FROM t")

    assert rows == [{"article_id": "a1", "score": 2.5}, {"article_id": "a2", "score": 1}]


async def test_the_caller_never_writes_the_format_clause():
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        return httpx.Response(200, text="")

    await client_over(handler).query_rows("SELECT 1;")

    assert sent[0] == "SELECT 1 FORMAT JSONEachRow"


async def test_the_database_is_sent_as_a_parameter():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params["database"])
        return httpx.Response(200, text="")

    await client_over(handler).query_rows("SELECT 1")

    # Lowercase `explained`: ClickHouse identifiers are case-sensitive and `explAIned` does
    # not exist on the deployed instance.
    assert seen == ["explained"]


async def test_an_empty_result_is_an_empty_list_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n")

    assert await client_over(handler).query_rows("SELECT 1") == []


async def test_a_server_error_is_raised_with_its_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Code: 60. DB::Exception: Table does not exist")

    with pytest.raises(ClickHouseError, match="Table does not exist"):
        await client_over(handler).query_rows("SELECT 1")


async def test_a_transport_failure_is_a_clickhouse_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ClickHouseError):
        await client_over(handler).query_rows("SELECT 1")


async def test_execute_sends_the_statement_unchanged():
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content.decode("utf-8"))
        return httpx.Response(200, text="")

    await client_over(handler).execute("CREATE TABLE t (a UInt8) ENGINE = Memory")

    assert sent[0].startswith("CREATE TABLE")
    assert "FORMAT" not in sent[0]
