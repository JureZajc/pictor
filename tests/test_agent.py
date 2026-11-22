import asyncio
import base64
import contextlib
import io

from PIL import Image

from pictor.agent import Agent, encode_for_model, explain_failure, limits_note, resolve_think
from pictor.config import Config
from pictor.ollama import OllamaError
from pictor.tools import ToolRegistry, ToolResult
from tests.conftest import FakeGenerator, FakeOllama, call, calls, say

MODEL = "gemma4:12b"


def make_agent(script, config: Config | None = None, **fake_options) -> tuple[Agent, FakeOllama]:
    fake = FakeOllama(script, **fake_options)
    agent = Agent(fake, config or Config(model=MODEL), budget_gb=16.8)
    return agent, fake


async def collect(agent: Agent, session, text: str) -> list[dict]:
    return [event async for event in agent.run(session, text)]


def kinds(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


def reply_text(events: list[dict]) -> str:
    return "".join(event["delta"] for event in events if event["type"] == "text")


# ---------------------------------------------------------------- plain answers


async def test_a_question_is_answered_without_touching_the_picture(session):
    agent, fake = make_agent([say("It is a red square on blue.", thinking="Let me look.")])
    events = await collect(agent, session, "What is in this picture?")

    assert kinds(events)[0] == "status"
    assert "thinking" in kinds(events)
    assert reply_text(events) == "It is a red square on blue."
    assert events[-1] == {"type": "done", "steps": 0, "limit_reached": False}
    assert session.cursor == 0

    (request,) = fake.requests
    system, user = request["messages"]
    assert system["role"] == "system" and "Pictor" in system["content"]
    assert "You can see it" in system["content"]
    assert user["content"].startswith("What is in this picture?")
    assert "640×480 px" in user["content"]
    assert len(user["images"]) == 1
    assert request["options"] == {"num_ctx": 16384}
    assert request["model"] == MODEL
    names = {tool["function"]["name"] for tool in request["tools"]}
    assert {"crop", "remove_background", "view_image"} <= names


async def test_the_picture_is_sent_scaled_down_as_jpeg(session):
    big = Image.new("RGB", (4000, 2000), "green")
    session.push(big, "Resize")
    agent, fake = make_agent([say("ok")])
    await collect(agent, session, "look")
    sent = Image.open(io.BytesIO(base64.b64decode(fake.requests[0]["messages"][1]["images"][0])))
    assert sent.format == "JPEG" and sent.size == (1024, 512)


def test_transparency_is_shown_to_the_model_as_a_checkerboard():
    clear = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    seen = Image.open(io.BytesIO(base64.b64decode(encode_for_model(clear, 1024))))
    assert seen.mode == "RGB"
    light, dark = seen.getpixel((4, 4)), seen.getpixel((20, 4))
    assert light != dark and light[0] > 200 and abs(dark[0] - dark[2]) < 12  # greys, not red


# ---------------------------------------------------------------- tool use


async def test_tools_are_run_and_their_results_fed_back(session):
    agent, fake = make_agent(
        [
            call("apply_filter", name="grayscale"),
            call("crop_to_aspect_ratio", ratio="1:1"),
            say("Done: black and white, square."),
        ]
    )
    events = await collect(agent, session, "Make it black and white and square")

    assert [v.label for v in session.versions] == ["Original", "Filter", "Crop to ratio"]
    assert session.image().size == (480, 480)
    r, g, b = session.image().getpixel((50, 50))
    assert r == g == b

    starts = [e for e in events if e["type"] == "tool_start"]
    ends = [e for e in events if e["type"] == "tool_end"]
    assert [e["name"] for e in starts] == ["apply_filter", "crop_to_aspect_ratio"]
    assert [e["label"] for e in starts] == ["Filter", "Crop to ratio"]
    assert starts[1]["args"] == {"ratio": "1:1"}
    assert all(e["ok"] for e in ends)
    assert ends[1]["version"] == {
        "id": 2,
        "label": "Crop to ratio",
        "detail": "1:1",
        "width": 480,
        "height": 480,
    }
    states = [e["state"] for e in events if e["type"] == "state"]
    assert [s["cursor"] for s in states] == [1, 2]
    assert events[-1] == {"type": "done", "steps": 2, "limit_reached": False}
    # start, end and state for one call arrive in that order
    first = kinds(events).index("tool_start")
    assert kinds(events)[first : first + 3] == ["tool_start", "tool_end", "state"]

    # The second request carries the first call and its result.
    second = fake.requests[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"]
    assert second[-1] == {
        "role": "tool",
        "tool_name": "apply_filter",
        "content": "Applied the grayscale filter at strength 100.",
    }


async def test_several_calls_in_one_turn_all_run(session):
    agent, fake = make_agent(
        [
            calls(("flip", {"direction": "horizontal"}), ("resize", {"percent": 50})),
            say("Mirrored and halved."),
        ]
    )
    events = await collect(agent, session, "mirror it and halve it")
    assert session.cursor == 2 and session.image().size == (320, 240)
    assert [m["role"] for m in fake.requests[1]["messages"][-3:]] == ["assistant", "tool", "tool"]
    assert events[-1]["steps"] == 2


async def test_mistakes_are_reported_to_the_model_so_it_can_recover(session):
    agent, fake = make_agent(
        [
            call("sharpen_everything", amount=5),  # no such tool
            call("flip", direction="sideways"),  # invalid argument
            call("crop", left=50, top=0, right=50, bottom=100),  # empty region
            call("flip", direction="vertical"),
            say("Flipped it."),
        ]
    )
    events = await collect(agent, session, "flip it upside down")

    ends = [e for e in events if e["type"] == "tool_end"]
    assert [e["ok"] for e in ends] == [False, False, False, True]
    assert "no tool named 'sharpen_everything'" in ends[0]["result"]
    assert "Available tools:" in ends[0]["result"] and "flip" in ends[0]["result"]
    assert ends[1]["result"].startswith("Error: Invalid arguments for flip")
    assert "empty" in ends[2]["result"]
    assert session.cursor == 1  # only the valid call made a version
    assert fake.requests[1]["messages"][-1]["content"] == ends[0]["result"]
    assert reply_text(events) == "Flipped it."


async def test_arguments_sent_as_a_json_string_are_accepted(session):
    turn = [
        {
            "message": {
                "tool_calls": [{"function": {"name": "resize", "arguments": '{"percent": 50}'}}]
            }
        },
        {"done": True},
    ]
    agent, _ = make_agent([turn, say("Halved.")])
    await collect(agent, session, "halve it")
    assert session.image().size == (320, 240)


async def test_the_same_edit_is_not_applied_twice_in_a_row(session):
    agent, _ = make_agent(
        [
            call("resize", percent=50),
            call("resize", percent=50),  # a small model repeating itself
            say("Halved."),
        ]
    )
    events = await collect(agent, session, "halve it")
    ends = [e for e in events if e["type"] == "tool_end"]
    assert session.image().size == (320, 240)
    assert ends[1]["ok"] is False and "has not been repeated" in ends[1]["result"]


async def test_a_repeat_is_allowed_after_something_else_happened(session):
    agent, _ = make_agent(
        [
            call("rotate", degrees=90),
            call("get_image_info"),
            call("rotate", degrees=90),
            say("Turned it upside down."),
        ]
    )
    await collect(agent, session, "turn it twice")
    assert session.cursor == 2 and session.image().size == (640, 480)


async def test_the_model_can_undo_its_own_edit(session):
    agent, _ = make_agent(
        [
            call("crop", left=0, top=0, right=10, bottom=10),
            call("undo_last_edit"),
            call("crop", left=0, top=0, right=50, bottom=50),
            say("Cropped to the top-left quarter."),
        ]
    )
    events = await collect(agent, session, "crop to the top-left quarter")
    assert [v.label for v in session.versions] == ["Original", "Crop"]
    assert session.image().size == (320, 240)
    cursors = [e["state"]["cursor"] for e in events if e["type"] == "state"]
    assert cursors == [1, 0, 1]


async def test_view_image_shows_the_model_the_current_picture(session):
    agent, fake = make_agent(
        [
            call("crop", left=0, top=0, right=50, bottom=50),
            call("view_image"),
            say("The crop shows the red area."),
        ]
    )
    await collect(agent, session, "crop to the red part and check")

    look = fake.requests[2]["messages"][-1]
    assert look["role"] == "user" and "320×240 px" in look["content"]
    seen = Image.open(io.BytesIO(base64.b64decode(look["images"][0])))
    assert seen.size == (320, 240)
    # The first request's picture is still the original, so the prompt prefix is stable.
    original = fake.requests[2]["messages"][1]["images"][0]
    assert Image.open(io.BytesIO(base64.b64decode(original))).size == (640, 480)


async def test_a_model_without_vision_gets_no_pictures(session):
    agent, fake = make_agent(
        [call("view_image"), say("I cannot see it.")],
        capabilities=("completion", "tools"),
    )
    events = await collect(agent, session, "what is this?")
    request = fake.requests[0]
    assert "images" not in request["messages"][1]
    assert "You cannot see pictures" in request["messages"][0]["content"]
    assert "view_image" not in {t["function"]["name"] for t in request["tools"]}
    # Asking for it anyway is answered like any unknown tool.
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] is False and "no tool named 'view_image'" in end["result"]
    assert all("images" not in m for m in fake.requests[1]["messages"])


async def test_slow_tools_can_report_progress(session):
    tools = ToolRegistry()

    @tools.tool("Slow")
    def slow_tool(ctx) -> ToolResult:
        """A tool that takes a while and says what it is doing."""
        ctx.status("Downloading something…")
        ctx.status("Nearly there…")
        return ToolResult("Finished.")

    fake = FakeOllama([call("slow_tool"), say("Done.")])
    agent = Agent(fake, Config(model=MODEL), budget_gb=8, tools=tools)
    events = await collect(agent, session, "do the slow thing")
    notes = [e["text"] for e in events if e["type"] == "tool_status"]
    assert notes == ["Downloading something…", "Nearly there…"]
    assert kinds(events).index("tool_status") < kinds(events).index("tool_end")
    assert fake.requests[0]["options"] == {"num_ctx": 12288}


async def test_a_crashing_tool_does_not_end_the_run(session):
    tools = ToolRegistry()

    @tools.tool("Broken")
    def broken_tool(ctx) -> ToolResult:
        """A tool with a bug."""
        raise RuntimeError("boom")

    fake = FakeOllama([call("broken_tool"), say("That did not work.")])
    agent = Agent(fake, Config(model=MODEL), budget_gb=8, tools=tools)
    events = await collect(agent, session, "try it")
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] is False and "failed unexpectedly (boom)" in end["result"]
    assert events[-1]["type"] == "done"


# ---------------------------------------------------------------- painting new things

DOG = {"description": "a small brown dog", "left": 55, "top": 10, "right": 95, "bottom": 45}


def painting_agent(script, generator, budget_gb: float = 16.8) -> tuple[Agent, FakeOllama]:
    fake = FakeOllama(script)
    return Agent(fake, Config(model=MODEL), budget_gb=budget_gb, generator=generator), fake


async def test_the_model_can_paint_something_new_when_a_generator_is_set_up(session, generator):
    agent, fake = painting_agent([call("add_object", **DOG), say("Added a dog.")], generator)
    events = await collect(agent, session, "add a small brown dog on the right")

    request = fake.requests[0]
    offered = {tool["function"]["name"] for tool in request["tools"]}
    assert {"add_object", "remove_object"} <= offered
    system = request["messages"][0]["content"]
    assert "call add_object" in system and "call remove_object" in system
    assert "such as generating new content" not in system

    assert [v.label for v in session.versions] == ["Original", "Add object"]
    assert session.image().size == (640, 480)
    start = next(e for e in events if e["type"] == "tool_start")
    assert (start["name"], start["label"], start["args"]) == ("add_object", "Add object", DOG)
    notes = [e["text"] for e in events if e["type"] == "tool_status"]
    assert notes == ["Loading the image model…", "Painting: step 1 of 4…"]
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] and end["result"].startswith("Painted a small brown dog into the region")
    assert end["version"]["label"] == "Add object"
    assert end["version"]["detail"] == "a small brown dog"
    assert kinds(events).count("state") == 1 and events[-1]["steps"] == 1
    # The model is shown the result, so it can judge what the image model painted.
    look = fake.requests[1]["messages"][-1]
    assert look["role"] == "user" and len(look["images"]) == 1
    assert generator.calls[0]["prompt"].startswith("Add a small brown dog to this picture")


async def test_without_a_generator_painting_is_not_offered_and_the_model_is_told_so(session):
    for generator in (None, FakeGenerator(state="no_weights"), FakeGenerator(state="unsupported")):
        agent, fake = painting_agent(
            [call("add_object", **DOG), say("I cannot do that.")], generator
        )
        events = await collect(agent, session, "add a dog")

        request = fake.requests[0]
        offered = {tool["function"]["name"] for tool in request["tools"]}
        assert "add_object" not in offered and "remove_object" not in offered
        assert "such as generating new content" in request["messages"][0]["content"]
        # A model that calls it anyway is answered like for any unknown tool.
        end = next(e for e in events if e["type"] == "tool_end")
        assert end["ok"] is False and "no tool named 'add_object'" in end["result"]
        assert "add_object" not in end["result"].split("Available tools:")[1]
        assert session.cursor == 0
        if generator is not None:
            assert generator.calls == []


def test_the_system_prompt_only_promises_what_the_tools_can_do():
    nothing = limits_note(["crop", "resize"])
    assert "cannot be done with your tools, such as generating new content" in nothing
    adding = limits_note(["crop", "add_object"])
    assert "call add_object" in adding and "remove_object" not in adding
    assert "Erasing an object" in adding  # still not possible without the removal tool
    # Where painting could be set up, the model points the user to the switch.
    hint = limits_note(["crop", "resize"], can_be_switched_on=True)
    assert "cannot be done with your tools" in hint and "'Add things to pictures'" in hint
    assert "Add things to pictures" not in nothing
    both = limits_note(["add_object", "remove_object"])
    assert "call remove_object" in both and "Erasing an object" not in both
    assert "repainting the whole picture" in both


async def test_the_language_model_steps_aside_when_both_do_not_fit_in_memory(session, generator):
    agent, fake = painting_agent([call("add_object", **DOG), say("Added a dog.")], generator)
    fake.in_memory = {MODEL: 11.1}  # plus 8.5 GB for the image model: over the 16.8 GB budget
    events = await collect(agent, session, "add a dog")

    assert fake.unloaded == [MODEL]
    notes = [e["text"] for e in events if e["type"] == "tool_status"]
    assert notes[0].startswith("Making room in memory") and "Painting: step 1 of 4…" in notes
    # Nothing reloads it by hand: the next request does, and the user is told why it waits.
    order = kinds(events)
    waking = [i for i, e in enumerate(events) if e.get("text") == "Waking the model again…"]
    assert len(waking) == 1 and order[waking[0]] == "status"
    assert order.index("tool_end") < waking[0] < order.index("text")
    assert session.cursor == 1


async def test_the_language_model_stays_loaded_when_there_is_room(session, generator):
    agent, fake = painting_agent(
        [call("add_object", **DOG), say("Added a dog.")], generator, budget_gb=33.6
    )
    fake.in_memory = {MODEL: 11.1}
    events = await collect(agent, session, "add a dog")
    assert fake.unloaded == [] and session.cursor == 1
    assert not any("Making room" in e.get("text", "") for e in events)
    assert not any(e.get("text") == "Waking the model again…" for e in events)


async def wait_for(condition, seconds: float = 3.0) -> bool:
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.01)
    return False


async def test_stopping_the_run_tells_a_tool_that_is_still_working(session):
    generator = FakeGenerator(wait_for_cancel=True)  # paints until it is told to stop
    agent, _ = painting_agent([call("add_object", **DOG)], generator)
    seen = []

    async def consume() -> None:
        async for event in agent.run(session, "add a dog"):
            seen.append(event["type"])

    task = asyncio.ensure_future(consume())
    assert await wait_for(generator.started.is_set)
    assert not generator.was_cancelled
    task.cancel()  # what the server does when the browser's Stop closes the stream
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert await wait_for(lambda: generator.was_cancelled)
    assert "tool_end" not in seen and session.cursor == 0
    assert session.turns == []  # a stopped exchange is not remembered


async def test_closing_the_event_stream_also_stops_the_tool(session):
    generator = FakeGenerator(wait_for_cancel=True)
    agent, _ = painting_agent([call("add_object", **DOG)], generator)
    stream = agent.run(session, "add a dog")
    async for event in stream:
        if event["type"] == "tool_status":
            break
    await stream.aclose()
    assert await wait_for(lambda: generator.was_cancelled)


# ---------------------------------------------------------------- limits and failures


async def test_the_step_limit_stops_a_model_that_never_finishes(session):
    config = Config(model=MODEL, max_steps=3)
    agent, fake = make_agent([call("get_image_info")] * 3, config)
    events = await collect(agent, session, "go")
    assert events[-1] == {"type": "done", "steps": 3, "limit_reached": True}
    assert "stopped after 3 steps" in reply_text(events)
    assert len(fake.requests) == 3
    assert session.turns[-1][-1]["content"].startswith("I stopped after 3 steps")


async def test_an_ollama_failure_ends_the_run_but_keeps_finished_edits(session):
    agent, _ = make_agent(
        [call("resize", percent=50), OllamaError("mlx runner failed: Insufficient Memory")]
    )
    events = await collect(agent, session, "halve it")
    assert events[-1]["type"] == "error"
    assert "does not fit in this computer's memory" in events[-1]["message"]
    assert session.cursor == 1  # the resize happened and stays
    assert session.turns == []  # but the broken exchange is not remembered


async def test_missing_model_and_models_without_tools_are_refused(session):
    agent, _ = make_agent([], Config(model=None))
    (event,) = await collect(agent, session, "hi")
    assert event["type"] == "error" and "No model" in event["message"]

    agent, fake = make_agent([], capabilities=("completion", "vision"))
    (event,) = await collect(agent, session, "hi")
    assert event["type"] == "error" and "cannot call tools" in event["message"]
    assert fake.requests == []


def test_failures_are_explained_in_terms_the_user_can_act_on():
    oom = explain_failure("mlx: [METAL] Command buffer execution failed: Insufficient Memory", "m")
    assert "Choose a smaller model" in oom and "Insufficient Memory" in oom
    assert "not installed" in explain_failure("model 'm' not found", "m")
    assert explain_failure("connection reset", "m") == "connection reset"


# ---------------------------------------------------------------- memory across turns


async def test_earlier_exchanges_are_remembered_without_pictures_or_reasoning(session):
    agent, fake = make_agent(
        [
            call("resize", percent=50),
            say("Halved it.", thinking="Halving means 50 percent."),
            say("You asked me to halve it."),
        ]
    )
    await collect(agent, session, "halve it")
    await collect(agent, session, "what did I ask before?")

    third = fake.requests[2]["messages"]
    assert [m["role"] for m in third] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    old_user, new_user = third[1], third[-1]
    assert "images" not in old_user and "[picture omitted]" in old_user["content"]
    assert all("thinking" not in m for m in third[:-1])
    assert third[2]["tool_calls"]  # the old call is still there for context
    assert "320×240 px" in new_user["content"] and len(new_user["images"]) == 1
    assert len(session.turns) == 2


async def test_only_recent_exchanges_are_sent(session):
    agent, fake = make_agent([say(f"answer {n}") for n in range(9)])
    for n in range(9):
        await collect(agent, session, f"question {n}")
    last = fake.requests[-1]["messages"]
    users = [m["content"] for m in last if m["role"] == "user"]
    assert len(users) == 7  # six remembered plus the new one
    assert users[0].startswith("question 2") and users[-1].startswith("question 8")


# ---------------------------------------------------------------- thinking control


def test_resolve_think():
    levels = {"capabilities": ["thinking"], "thinking": {"values": [False, "low", "medium"]}}
    on_off = {"capabilities": ["thinking"], "thinking": {"values": [True, False]}}
    plain = {"capabilities": ["completion", "tools"]}

    assert resolve_think("auto", levels) == "low"
    assert resolve_think("auto", on_off) is None  # leave the model's default
    assert resolve_think("off", levels) is False
    assert resolve_think("off", on_off) is False
    assert resolve_think("default", levels) is None
    for mode in ("auto", "off", "default"):
        assert resolve_think(mode, plain) is None  # nothing to control


async def test_the_think_setting_reaches_the_model(session):
    thinking = {"values": [False, "low", "medium", "xhigh"], "default": "medium"}
    agent, fake = make_agent([say("hi")], thinking=thinking)
    await collect(agent, session, "hi")
    assert fake.requests[0]["think"] == "low"

    agent, fake = make_agent([say("hi")], Config(model=MODEL, think="off"), thinking=thinking)
    await collect(agent, session, "hi")
    assert fake.requests[0]["think"] is False
