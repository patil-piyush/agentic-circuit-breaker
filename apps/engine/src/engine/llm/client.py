"""
LLM client factory.

Returns a LangChain-compatible chat model based on LLM_PROVIDER config:
  - "openai"    → ChatOpenAI (gpt-4o-mini by default)
  - "anthropic" → ChatAnthropic (claude-haiku by default)
  - "mock"      → MockChatModel (deterministic, zero API cost — default for dev)

Usage:
    from engine.llm.client import get_llm
    llm = get_llm()
    response = llm.invoke([HumanMessage(content="Hello")])
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from engine.config import get_settings

# ── Adversarial query signals (queries the mock will treat as unanswerable) ────
# These phrases indicate hyper-specific data that the corpus cannot supply.
_ADVERSARIAL_SIGNALS: tuple[str, ...] = (
    "per country per year",
    "per country",
    "exact co2 reduction",
    "exact co₂ reduction",
    "broken down by sector",
    "tonnage figures",
    "per manufacturer per quarter",
    "proprietary r&d",
    "patent filing counts",
    "gdp impact coefficient",  # singular
    "gdp impact coefficients",  # plural
    "statistical significance values",
    "confidence intervals for each",
    "precise gdp",
    "per quarter from q1",
    "monthly granularity",
    "every country in the world",
    "every agricultural subsidy",
    "exact number",
    "exact figure",
    # Expanded dataset signals (FA2)
    "per solar farm per day",
    "exact gps coordinates",
    "blade failure rates per manufacturer",
    "warranty claim amounts",
    "repair duration histograms",
    "every turbine installed",
    "every noaa monitoring station",
    "sensor calibration offsets",
    "every 6 hours",
    "proprietary chemical formulas",
    "synthesis temperatures",
    "coating compound",
    "carbon credit price per tonne",
    "every transaction",
    "voluntary carbon market",
    "buyer, seller, and registry",
    "registry id",
    "turbine revolutions per minute",
    "every dam globally",
    "maintenance downtime logs",
    "methane emission readings",
    "parts per billion",
    "every oil and gas well",
    "measured daily from",
)

# Topics the corpus CAN answer (general knowledge docs are present)
_ANSWERABLE_TOPICS: tuple[str, ...] = (
    "solar energy",
    "photovoltaic",
    "solar panel",
    "solar system",
    "planet",
    "planets",
    "sun",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "astronomy",
    "climate change",
    "greenhouse gas",
    "co2",
    "renewable energy",
    "wind energy",
    "wind turbine",
    "hydropower",
    "circuit breaker",
    "multi-agent",
    "retrieval-augmented",
    "rag",
    "langgraph",
    "thrashing",
    "recursion",
)


def _is_adversarial(text: str) -> bool:
    """
    Returns True if the text contains signals of an unanswerable (adversarial) query.
    Adversarial queries ask for hyper-specific data the corpus cannot supply.
    """
    lower = text.lower()
    return any(signal in lower for signal in _ADVERSARIAL_SIGNALS)


def _is_answerable(text: str) -> bool:
    """Returns True if the text maps to a topic the corpus can address."""
    lower = text.lower()
    return any(topic in lower for topic in _ANSWERABLE_TOPICS)


# ── Mock LLM (zero-cost, deterministic) ──────────────────────────────────────


class MockChatModel(BaseChatModel):
    """
    A deterministic fake LLM for development and unit testing.

    Simulates real agent behavior without any API calls:
      - Researcher: returns topic-relevant notes for answerable queries;
        returns "couldn't find specific data" for adversarial queries.
      - Critic: returns SATISFIED when research notes contain topic content;
        returns RETRY when research notes indicate failure (drives the loop).
      - Writer: composes a realistic, structured answer (or graceful partial summary).

    This gives the correct behavior for both scenarios:
      - Normal queries → completes (status: "completed")
      - Adversarial queries → crashes (status: "crashed", GraphRecursionError)
    """

    model_name: str = "mock"

    # ── Researcher: success responses (used when corpus has relevant content) ──
    _researcher_success_template: str = (
        "Based on the corpus retrieval, I found relevant information about this topic.\n\n"
        "{topic_content}\n\n"
        "The corpus documents provide sufficient background to address this query. "
        "Key facts have been identified and are ready for the Critic's evaluation."
    )

    # ── Researcher: failure responses (adversarial — corpus lacks specific data) ──
    # IMPORTANT: These responses must be semantically near-identical so that the
    # cosine similarity between consecutive embeddings rises above the HALF_OPEN
    # threshold (0.85). The corpus does not contain the requested specific data.
    # Slight wording variation is intentional to prevent exact string matching
    # while keeping the semantic meaning (and thus the embedding) nearly identical.
    _researcher_failure_responses: tuple[str, ...] = (
        (
            "The corpus does not contain the specific quantitative data needed "
            "to answer this query. After thorough retrieval, the available documents "
            "lack the exact figures and granular statistics requested. "
            "The corpus cannot provide this information."
        ),
        (
            "The corpus does not contain the specific quantitative data required "
            "to answer this query. After thorough retrieval, the available documents "
            "lack the exact figures and granular statistics requested. "
            "The corpus cannot supply this information."
        ),
        (
            "The corpus does not contain the specific quantitative data necessary "
            "to answer this query. After thorough retrieval, the available documents "
            "lack the exact figures and granular statistics requested. "
            "The corpus cannot provide these details."
        ),
    )
    _failure_index: int = 0  # cycles through failure responses deterministically

    # ── Critic responses ──────────────────────────────────────────────────────
    _critic_retry: str = (
        "RETRY: The research is insufficient. The retrieved information does not "
        "contain the specific quantitative data required to answer the query. "
        "Please search again with a more targeted approach."
    )
    _critic_satisfied: str = (
        "SATISFIED: The research provides adequate information to draft an answer. "
        "The key facts have been retrieved from the corpus and the findings are "
        "comprehensive enough to compose a final response."
    )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._pick_response(messages)
        token_in = self._count_input_tokens(messages)
        token_out = len(content.split())
        message = AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": token_in,
                "output_tokens": token_out,
                "total_tokens": token_in + token_out,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _pick_response(self, messages: list[BaseMessage]) -> str:
        """
        Route to the correct canned response based on:
          1. Which agent is calling (system message role hint)
          2. Whether the query is adversarial or answerable (message content analysis)

        Role detection uses 'you are a X agent' to avoid false matches
        (e.g., the critic's system prompt mentions 'Researcher' by name).
        """
        system_content = ""
        human_content = ""

        for msg in messages:
            if msg.type == "system":
                system_content = str(msg.content).lower()
            elif msg.type == "human":
                human_content = str(msg.content)

        # Use unambiguous role detection — check "you are a X agent" pattern
        if "you are a researcher agent" in system_content:
            return self._researcher_response(human_content)
        elif "you are a critic agent" in system_content:
            return self._critic_response(human_content)
        elif "you are a writer agent" in system_content:
            return self._writer_response(system_content, human_content)
        else:
            return "I have processed your request and generated a response."

    def _writer_response(self, system_content: str, human_content: str) -> str:
        """
        Synthesize a realistic, query-specific final answer or forced partial summary.
        """
        lower_sys = system_content.lower()
        lower_human = human_content.lower()

        # Check if Writer was triggered under forced summarization / circuit breaker open
        is_partial = (
            "partial answer" in lower_sys
            or "partial — corpus could not fully answer" in lower_human
            or "circuit breaker" in lower_human
            or _is_adversarial(human_content)
        )

        if is_partial:
            return self._compose_partial_summary(human_content)

        return self._compose_final_answer(human_content)

    def _compose_partial_summary(self, human_content: str) -> str:
        """Compose a structured, honest partial answer when the circuit breaker trips."""
        query = "the requested topic"
        for line in human_content.splitlines():
            if line.strip().lower().startswith("query:"):
                query = line.split(":", 1)[1].strip()
                break

        return (
            "### [Partial Answer — Circuit Breaker Intervention]\n\n"
            "**Notice:** The autonomous agentic loop was halted by the Agentic Circuit Breaker after detecting "
            "semantic thrashing (repeated retrieval cycles without gaining new information). The document corpus "
            "does not contain the complete granular dataset required to fully answer this query.\n\n"
            "**1. Findings from Available Knowledge Base:**\n"
            "• The knowledge base contains general background on climate change, greenhouse gas concentrations, "
            "and global energy transition trends.\n"
            "• Atmospheric CO2 concentrations reached approximately 421 ppm in 2023, with global average "
            "temperatures 1.1°C above pre-industrial baselines.\n"
            "• Transitioning to renewable energy (solar, wind, hydro) is recognized as the key lever to achieve "
            "net-zero emissions by 2050.\n\n"
            "**2. Missing Information (Corpus Limitations):**\n"
            f'• Detailed per-country, per-year quantitative metrics requested in: "{query}".\n'
            "• Sector-by-sector breakdowns (energy, transport, agriculture, industry) for each individual UN member state.\n\n"
            "**3. Recommended External Sources:**\n"
            "• UNFCCC National Greenhouse Gas Inventory Submissions (Common Reporting Format tables)\n"
            "• International Energy Agency (IEA) World Energy Balances & Emissions Data\n"
            "• European Commission EDGAR (Emissions Database for Global Atmospheric Research)"
        )

    def _compose_final_answer(self, human_content: str) -> str:
        """Compose a realistic, topic-tailored answer for answerable queries."""
        lower = human_content.lower()

        if "solar system" in lower or "planet" in lower or "planets" in lower:
            return (
                "### The Solar System: Architecture and Planetary Bodies\n\n"
                "The Solar System is gravitationally bound to the Sun and formed approximately 4.6 billion years ago "
                "from the gravitational collapse of a giant interstellar molecular cloud.\n\n"
                "**1. The Central Star:**\n"
                "• **The Sun:** A G-type main-sequence star (yellow dwarf) containing ~99.86% of the system's total mass. "
                "Its gravitational pull dictates the orbits of all planets, asteroids, and comets.\n\n"
                "**2. The Eight Planets (in orbital order from the Sun):**\n"
                "• **Mercury:** Smallest planet, terrestrial, with extreme temperature variations and no substantial atmosphere.\n"
                "• **Venus:** Earth-sized terrestrial planet with a runaway greenhouse effect and thick toxic CO2 atmosphere.\n"
                "• **Earth:** Only known planet to harbor life, with liquid surface water and a protective nitrogen-oxygen atmosphere.\n"
                "• **Mars:** The 'Red Planet', possessing polar ice caps, ancient river valleys, and a thin carbon dioxide atmosphere.\n"
                "• **Jupiter:** Largest planet, a gas giant dominated by hydrogen and helium with the iconic Great Red Spot.\n"
                "• **Saturn:** Second-largest gas giant, distinguished by an extensive, complex system of icy rings.\n"
                "• **Uranus:** An ice giant with methane in its atmosphere, known for its unique 98-degree axial tilt.\n"
                "• **Neptune:** Outermost ice giant, characterized by supersonic winds and active weather storms.\n\n"
                "**3. Additional Planetary Features:**\n"
                "• **Asteroid Belt:** Situated between Mars and Jupiter, containing rocky remnants and dwarf planet Ceres.\n"
                "• **Kuiper Belt & Oort Cloud:** Vast outer regions containing icy bodies, comets, and dwarf planets (including Pluto, Eris, Haumea, and Makemake)."
            )

        if "solar" in lower or "photovoltaic" in lower:
            return (
                "### Solar Energy: Fundamentals and Working Principles\n\n"
                "Solar energy is radiant light and heat from the Sun harnessed using technologies such as "
                "photovoltaics (PV), concentrated solar power (CSP), and solar water heating. It represents "
                "an abundant, clean, and renewable energy source.\n\n"
                "**1. How Photovoltaic (PV) Panels Work:**\n"
                "• **Photovoltaic Effect:** Solar panels consist of semiconductor cells, primarily silicon. "
                "When sunlight (photons) strikes the semiconductor material, it knocks electrons free, "
                "producing a flow of direct current (DC) electricity.\n"
                "• **Inversion to AC:** An inverter converts DC electricity into alternating current (AC), "
                "enabling it to power commercial electrical appliances and connect to the public electrical grid.\n"
                "• **Panel Efficiency:** Modern commercial monocrystalline panels reach 20%–22%+ efficiency, "
                "while polycrystalline models typically achieve 15%–18%.\n\n"
                "**2. Global Capacity & Economic Impact:**\n"
                "• Global installed solar PV capacity has exceeded 1.6 terawatts (TW), generating over 1,600 TWh annually.\n"
                "• Due to technological scaling and manufacturing advancements, the levelized cost of energy (LCOE) "
                "for utility-scale solar has dropped by ~90% over the past decade (reaching ~$0.049/kWh), making it "
                "the cheapest source of new electricity generation in most regions worldwide."
            )

        if (
            "climate change" in lower
            or "greenhouse gas" in lower
            or "warming" in lower
            or "co2" in lower
        ):
            return (
                "### Climate Change: Causes, Impacts, and Solutions\n\n"
                "**1. Core Drivers:**\n"
                "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, "
                "chiefly the combustion of fossil fuels (coal, oil, gas) and deforestation, are the dominant cause of "
                "observed warming since the mid-20th century.\n\n"
                "**2. The Greenhouse Effect & Atmospheric Concentrations:**\n"
                "• Greenhouse gases (CO2, CH4, N2O) trap outgoing infrared radiation in the atmosphere.\n"
                "• Atmospheric CO2 concentrations stand at ~421 ppm (compared to pre-industrial 280 ppm).\n"
                "• Global average surface temperature has risen by ~1.1°C above pre-industrial levels, with the last "
                "decade being the hottest on record.\n\n"
                "**3. Pathways to Mitigation:**\n"
                "• Limiting warming to 1.5°C requires achieving global net-zero CO2 emissions by approximately 2050.\n"
                "• Decarbonizing energy via solar, wind, and storage is the fastest and most cost-effective lever available."
            )

        if "multi-agent" in lower or "circuit breaker" in lower or "rag" in lower:
            return (
                "### Multi-Agent Systems and Semantic Circuit Breakers\n\n"
                "**1. Multi-Agent RAG Architectures:**\n"
                "Multi-agent pipelines decompose complex reasoning tasks into specialized roles (e.g. Researcher, Critic, Writer). "
                "Retrieval-Augmented Generation (RAG) grounds agent responses in external document corpora to mitigate hallucinations.\n\n"
                "**2. Failure Modes & Semantic Thrashing:**\n"
                "When queries exceed the knowledge base, iterative critique loops often degenerate into semantic thrashing — "
                "agents reformulate identical queries, repeatedly querying without gaining new information until hitting "
                "hard recursion limits (GraphRecursionError).\n\n"
                "**3. Circuit Breaker Protection:**\n"
                "The Agentic Circuit Breaker tracks state vectors across agent turns. When cosine similarity between successive "
                "iterations exceeds threshold (≥ 0.95), the breaker transitions to OPEN, isolates the thrashing agent, and routes "
                "the pipeline to forced summarization, ensuring graceful degradation and preventing token exhaustion."
            )

        if "renewable energy" in lower or "wind" in lower or "hydropower" in lower:
            return (
                "### Renewable Energy: Technologies and Global Transition\n\n"
                "Renewable energy harnesses naturally replenishing resources that produce minimal greenhouse gas emissions:\n"
                "• **Solar Energy:** Photovoltaic and concentrated solar power harnessing solar irradiance.\n"
                "• **Wind Energy:** Onshore and offshore wind turbines converting aerodynamic kinetic energy to electricity.\n"
                "• **Hydropower:** Gravitational flow of water driving hydroelectric turbines, providing dispatchable baseload.\n"
                "Renewables now form the cornerstone of global decarbonization, driven by falling capital costs and grid-scale storage."
            )

        # Fallback to corpus snippet
        corpus_snippet = self._extract_corpus_snippet(human_content)
        return (
            "### Synthesized Research Findings\n\n"
            "Based on the corpus retrieval and agent evaluation, here is the synthesized answer:\n\n"
            f"{corpus_snippet}\n\n"
            "The retrieved evidence provides clear support for the core concepts of the query. "
            "All findings have been verified against the available corpus documents."
        )

    def _researcher_response(self, human_content: str) -> str:
        """
        Return a success response if the query is answerable,
        or cycle through failure responses if it's adversarial.
        """
        if _is_adversarial(human_content):
            # Cycle deterministically through failure responses
            idx = self._failure_index % len(self._researcher_failure_responses)
            object.__setattr__(self, "_failure_index", self._failure_index + 1)
            return self._researcher_failure_responses[idx]

        # Answerable query → extract any corpus content from the human prompt
        # and return a success response
        topic_snippet = self._extract_corpus_snippet(human_content)
        return self._researcher_success_template.format(topic_content=topic_snippet)

    def _critic_response(self, human_content: str) -> str:
        """
        Return SATISFIED if the research notes look adequate (no failure phrases),
        or RETRY if the research notes indicate the corpus couldn't answer.
        """
        # Failure phrases that the researcher emits for adversarial queries
        failure_phrases = (
            "corpus does not contain the specific quantitative data",
            "lack the exact figures",
            "corpus cannot provide",
            "corpus cannot supply",
            "could not locate specific",
            "exact numbers requested are not present",
            "specific data points needed",
            "lack precise figures",
            "not available in the current document set",
        )
        lower = human_content.lower()
        if any(phrase in lower for phrase in failure_phrases):
            return self._critic_retry
        return self._critic_satisfied

    def _extract_corpus_snippet(self, human_content: str) -> str:
        """
        Pull out the corpus documents section from the researcher's human prompt.
        This gives the mock writer something realistic to reference.
        """
        if "Available corpus documents:" in human_content:
            parts = human_content.split("Available corpus documents:")
            if len(parts) > 1:
                corpus_part = parts[1].strip()
                # Take first 400 chars of corpus content
                return corpus_part[:400] + "..." if len(corpus_part) > 400 else corpus_part
        return "The corpus documents provide relevant background information on this topic."

    def _count_input_tokens(self, messages: list[BaseMessage]) -> int:
        total_chars = sum(len(str(m.content)) for m in messages)
        return max(1, total_chars // 4)

    @property
    def _llm_type(self) -> str:
        return "mock"


# ── Real LLM Factories ────────────────────────────────────────────────────────


def _build_openai_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
            "Add it to your .env or use LLM_PROVIDER=mock for development."
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
        temperature=0.7,
    )


def _build_anthropic_llm() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
            "Add it to your .env or use LLM_PROVIDER=mock for development."
        )
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        temperature=0.7,
        max_tokens=1024,
    )


# ── Public API ────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return the cached LLM singleton based on LLM_PROVIDER config.

    - "mock"      → MockChatModel (default, free)
    - "openai"    → ChatOpenAI
    - "anthropic" → ChatAnthropic
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "mock":
        return MockChatModel()
    elif provider == "openai":
        return _build_openai_llm()
    elif provider == "anthropic":
        return _build_anthropic_llm()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Must be one of: 'mock', 'openai', 'anthropic'."
        )
