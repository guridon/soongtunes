import asyncio

from stunes.conversation.extract import SlotExtractor


class _LLM:
    def __init__(self):
        self.messages = None

    async def complete_json(self, messages, schema, name):
        self.messages = messages
        return {"intent": "recommend", "genres": ["인디"]}


def _run(ex, *args):
    return asyncio.run(ex.extract(*args))


def test_state_context_puts_the_stored_state_and_results_before_the_history():
    llm = _LLM()
    ex = SlotExtractor(llm, ["인디"], state_context=True)
    history = [{"role": "user", "content": "인디 추천해줘"}]
    slots, _ = _run(ex, "포크는 빼줘", history, "장르: 인디", "1. 곡 - 가수 (장르: 인디)")
    last = llm.messages[-1]["content"]
    assert last.index("[현재 조건]\n장르: 인디") < last.index("[직전 추천 목록]\n1. 곡")
    assert last.index("[직전 추천 목록]") < last.index("[직전 대화]\nuser: 인디 추천해줘")
    assert last.endswith("[최신 발화]\n포크는 빼줘")
    # the few-shot examples carry an empty state, so they don't read as this conversation
    assert llm.messages[1]["content"].startswith("[현재 조건]\n(없음)\n\n[직전 추천 목록]\n(없음)")
    assert slots.genres == ["인디"]


def test_cumulative_framing_is_unchanged_by_the_state_context_mode():
    llm = _LLM()
    _run(SlotExtractor(llm, ["인디"], cumulative=True), "더 보여줘", [])
    assert llm.messages[-1]["content"] == "[직전 대화]\n(없음)\n\n[최신 발화]\n더 보여줘"
    assert "[현재 조건]" not in llm.messages[0]["content"]


class _SchemaLLM(_LLM):
    def __init__(self, out):
        super().__init__()
        self.out, self.schema = out, None

    async def complete_json(self, messages, schema, name):
        self.messages, self.schema = messages, schema
        return self.out


def test_split_asks_for_the_new_topic_flag_and_moves_genre_exclusions():
    llm = _SchemaLLM(
        {
            "intent": "recommend",
            "genres": ["댄스"],
            "exclude_moods": ["아이돌", "새벽"],
            "exclude_artists": ["아이돌"],
            "new_topic": True,
        }
    )
    ex = SlotExtractor(llm, ["댄스", "아이돌"], split=True)
    slots, raw = _run(ex, "아이돌 말고 다른 걸로", [], "장르: 댄스", "")
    assert "new_topic" in llm.schema["required"]
    assert "new_topic" in llm.messages[0]["content"]
    assert '"new_topic":false' in llm.messages[2]["content"]  # the few-shot answers carry it
    assert "[현재 조건]\n장르: 댄스" in llm.messages[-1]["content"]
    assert slots.new_topic is True
    assert slots.exclude_genres == ["아이돌"]
    assert (slots.exclude_moods, slots.exclude_artists) == (["새벽"], [])
    assert raw["exclude_moods"] == ["아이돌", "새벽"]  # the trace keeps what the LLM said


def test_state_context_without_split_keeps_the_served_schema():
    llm = _SchemaLLM({"intent": "recommend", "exclude_moods": ["아이돌"]})
    slots, _ = _run(SlotExtractor(llm, ["아이돌"], state_context=True), "아이돌 말고", [])
    assert "new_topic" not in llm.schema["properties"]
    assert "new_topic" not in llm.messages[0]["content"]
    assert slots.exclude_moods == ["아이돌"]
