"""Entity-aware, deterministic query decomposition helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Sequence

from src.query.entity_extractor import EntityMode, NamedEntity, fold_entity_text
from src.query.spatial_types import SpatialRelationEnum

QUERY_DECOMPOSER_VERSION = "entity-query-decomposer-v8-color-subject-binding"

QueryIntent = Literal["visual_dense", "lexical_dominant", "spatial_color"]
SemanticViewRole = Literal[
    "global",
    "action",
    "scene_action",
    "foreground_subject",
    "distinctive_attributes",
    "instance_attributes",
    "relation_relaxed",
]
SpatialZone = Literal["left", "right", "center", "top", "bottom"]
SpatialRelation = Literal[
    "left_of", "right_of", "above", "below", "near", "far_from",
    "inside", "surrounding", "overlapping", "between", "in_front_of", "behind",
]

_COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("red", "đỏ"),
    "blue": ("blue", "xanh dương", "xanh da trời", "xanh duong"),
    "yellow": ("yellow", "vàng"),
    "green": ("green", "xanh lá", "xanh lục"),
    "black": ("black", "đen"),
    "white": ("white", "trắng"),
    "pink": ("pink", "hồng"),
    "purple": ("purple", "violet", "tím"),
}
_ZONE_ALIASES: dict[str, tuple[str, ...]] = {
    "left": ("on the left", "left side", "left", "bên trái", "phía trái"),
    "right": ("on the right", "right side", "right", "bên phải", "phía phải"),
    "center": (
        "in the center", "at the center", "middle", "center",
        "ở giữa", "chính giữa", "trung tâm",
    ),
    "top": (
        "at the top", "upper", "top", "phía trên", "bên trên", "ở trên",
        "background", "hậu cảnh",
    ),
    "bottom": (
        "at the bottom", "lower", "bottom", "phía dưới", "bên dưới", "ở dưới",
        "foreground", "tiền cảnh",
    ),
}
_LEXICAL_CUES = re.compile(
    r"\b(?:ocr|asr|subtitle|caption|logo|sign|banner|text|word|number|"
    r"biển\s+báo|bảng\s+chữ|phụ\s+đề|dòng\s+chữ|chữ|số|logo)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<!\w)\d+(?!\w)")
_QUOTED = re.compile(r"[\"“”'‘’][^\"“”'‘’]{1,}[\"“”'‘’]")
_PARENTHETICAL = re.compile(r"\((?P<text>[^()]{1,240})\)")
_VIETNAMESE_MARKED = frozenset(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)
_ACTION_BOUNDARY = re.compile(
    r"\b(?:wrapped|filled|containing|garnished|wearing|surrounded|followed|"
    r"prepare(?:s|d|ing)?|plan(?:s|ned|ning)?|wait(?:s|ed|ing)?|"
    r"cross(?:es|ed|ing)?|queue(?:s|d|ing)?|"
    r"được\s+bọc|chứa|trang\s+trí|mặc|bao\s+quanh|sau\s+đó)\b",
    re.IGNORECASE,
)
_SEQUENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bone\s+(?:after|behind)\s+another\b", re.IGNORECASE),
    re.compile(
        r"\b(?P<label>[^\W_]+)\s+after\s+(?P=label)\b",
        re.IGNORECASE | re.UNICODE,
    ),
    re.compile(r"(?<!\w)(?:lần\s+lượt|nối\s+đuôi\s+nhau)(?!\w)", re.IGNORECASE),
)
_PROSPECTIVE_RELATION = re.compile(
    r"\b(?P<verb>prepare(?:s|d|ing)?|plan(?:s|ned|ning)?|wait(?:s|ed|ing)?)\b"
    r"(?P<middle>[^,.;:]{0,96}?)\bto\s+"
    r"(?P<action>cross|enter|leave|pass|drive|move|walk|run)\s+"
    r"(?P<anchor>[^,.;:]{1,80})",
    re.IGNORECASE,
)
_TERNARY_RELATION_PATTERNS: tuple[tuple[SpatialRelation, re.Pattern[str]], ...] = (
    ("between", re.compile(
        r"(?P<b>[^,.;:]{1,120}?)\s+(?:is\s+|are\s+)?"
        r"(?:between|in-between|flanked\s+by)\s+(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+"
        r"(?:and|&)\s+(?P<c>[^,.;:]{1,120})", re.IGNORECASE,
    )),
    ("between", re.compile(
        r"(?P<b>[^,.;:]{1,120}?)\s+(?:(?:ở|nằm|kẹp|xen\s+kẽ)\s+)?giữa\s+"
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:và|với|&)\s+(?P<c>[^,.;:]{1,120})",
        re.IGNORECASE,
    )),
)
_BINARY_RELATION_PATTERNS: tuple[tuple[SpatialRelation, re.Pattern[str]], ...] = (
    ("overlapping", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?"
        r"(?:overlapping|overlaps|stacked\s+on|covering)\s+(?P<b>[^,.;:]{1,120})",
        re.IGNORECASE,
    )),
    ("overlapping", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:đè\s+lên|chồng\s+lên|gối\s+lên|trùng\s+lên|phủ\s+lên)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("surrounding", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?"
        r"(?:surrounding|surrounds|around|encircling|bordering|contains|holding|holds)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("surrounding", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:bao\s+quanh|xung\s+quanh|vây\s+quanh|viền\s+quanh|chứa|đựng)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("inside", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?"
        r"(?:inside|within|contained\s+in|in\s+the\s+middle\s+of)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("inside", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:ở\s+)?(?:bên\s+trong|nằm\s+trong|chứa\s+trong)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("left_of", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?(?:to\s+the\s+|on\s+the\s+)?left\s+of\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("left_of", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:(?:ở|nằm)\s+)?(?:bên|phía)\s+trái(?:\s+của)?\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("right_of", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?(?:to\s+the\s+|on\s+the\s+)?right\s+of\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("right_of", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:(?:ở|nằm)\s+)?(?:bên|phía)\s+phải(?:\s+của)?\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("above", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?(?:above|over|on\s+top\s+of|overhead\s+of)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("above", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:(?:ở|nằm)\s+)?(?:phía\s+trên|bên\s+trên|trên\s+đầu|trên)(?:\s+của)?\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("below", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?(?:below|under|underneath|beneath)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("below", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:(?:ở|nằm)\s+)?(?:phía\s+dưới|bên\s+dưới|dưới)(?:\s+của)?\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("near", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?(?:near|close\s+to|adjacent\s+to|next\s+to|beside|lined\s+up\s+near)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("near", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:(?:ở|nằm|xếp)\s+)?(?:gần|kế\s+bên|sát\s+bên|bên\s+cạnh|nằm\s+cạnh)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("far_from", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?(?:far\s+from|away\s+from|distant\s+from)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("far_from", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:(?:ở|nằm)\s+)?(?:xa|cách\s+xa|rời\s+xa)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("in_front_of", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?(?:in\s+front\s+of|in\s+the\s+foreground\s+of|ahead\s+of|leading)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("in_front_of", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:(?:ở|đứng|chạy)\s+)?(?:phía\s+trước|trước|vượt\s+trước)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("behind", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:is\s+|are\s+)?(?:behind|in\s+the\s+background\s+of|trailing|following)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
    ("behind", re.compile(
        r"(?P<a>(?:(?!\s+(?:and|và)\s+)[^,.;:]){1,120}?)\s+(?:(?:ở|đứng|chạy)\s+)?(?:phía\s+sau|đằng\s+sau|sau)\s+"
        r"(?P<b>[^,.;:]{1,120}?)(?=\s+(?:and|và)\s+|$|[,.;:])", re.IGNORECASE,
    )),
)
_AMBIGUOUS_BETWEEN = re.compile(
    r"(?<!\w)(?:among|amidst|ở\s+giữa|nằm\s+giữa|kẹp\s+giữa|xen\s+kẽ\s+giữa)(?!\w)",
    re.IGNORECASE,
)
_NEGATION_PATTERN = re.compile(
    r"(?<!\w)(?:no|without|not\s+containing|does\s+not\s+contain|"
    r"do\s+not\s+show|không\s+(?:có|chứa|xuất\s+hiện|thấy))"
    r"\s+(?P<subject>[^,.;:]{1,100})",
    re.IGNORECASE,
)

_SUBJECT_STOP = frozenset(
    {
        "a", "an", "the", "of", "on", "in", "at", "to", "with", "and", "or",
        "một", "những", "các", "của", "ở", "tại", "trong", "trên", "với", "và",
        "màu", "color", "coloured", "colored", "phía", "bên",
        "left", "right", "center", "middle", "top", "bottom", "upper", "lower",
        "trái", "phải", "giữa", "trên", "dưới", "trung", "tâm",
    }
)


@dataclass(frozen=True, slots=True)
class SpatialConstraint:
    subject: str | None
    zone: SpatialZone
    source_text: str
    perspective_proxy: bool = False

    @property
    def bound(self) -> bool:
        return bool(self.subject)


@dataclass(frozen=True, slots=True)
class ColorConstraint:
    subject: str | None
    color: str
    source_text: str
    node_id: str | None = None

    @property
    def bound(self) -> bool:
        return bool(self.subject)


@dataclass(frozen=True, slots=True)
class ObjectNode:
    """One deterministic object role referenced by verification constraints."""

    node_id: str
    label: str
    source_text: str


@dataclass(frozen=True, slots=True)
class ObjectInstance:
    """One separately verifiable mention within a repeated object group."""

    instance_id: str
    label: str
    canonical_label: str
    attributes: tuple[str, ...]
    group_id: str
    ordinal: int
    source_text: str


@dataclass(frozen=True, slots=True)
class SequenceConstraint:
    """An ordering idiom that must not be mistaken for a before/after split."""

    kind: Literal["ordered_group"]
    source_text: str
    member_label: str | None = None
    prospective_action: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class ProspectiveRelation:
    """An intended future action whose current spatial state is unspecified."""

    action: str
    anchor: str
    source_text: str
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class RelativeSpatialConstraint:
    """A binary or ternary relation over normalized 2-D object boxes."""

    subject: str | None
    relation: SpatialRelation
    object_subject: str | None
    source_text: str
    subject_node_id: str
    object_node_id: str
    second_object_subject: str | None = None
    second_object_node_id: str | None = None

    def __post_init__(self) -> None:
        aliases = {
            "adjacent", "adjacent_to", "contained_in", "contains", "around",
            "foreground", "background", "ahead_of",
        }
        if self.relation not in aliases:
            try:
                SpatialRelationEnum(self.relation)
            except ValueError as error:
                raise ValueError("unsupported spatial relation") from error

    @property
    def arity(self) -> int:
        return 3 if self.relation == SpatialRelationEnum.BETWEEN.value else 2

    @property
    def verification_type(self) -> str:
        return (
            "heuristic_2d"
            if self.relation in {
                SpatialRelationEnum.IN_FRONT_OF.value,
                SpatialRelationEnum.BEHIND.value,
            }
            else "geometry"
        )

    @property
    def bound(self) -> bool:
        binary = bool(self.subject and self.object_subject)
        return binary and (
            self.relation != SpatialRelationEnum.BETWEEN.value
            or bool(self.second_object_subject and self.second_object_node_id)
        )

@dataclass(frozen=True, slots=True)
class NegationConstraint:
    """An object, or object-color pair, that should not be present."""

    subject: str | None
    source_text: str
    color: str | None = None

    @property
    def bound(self) -> bool:
        return bool(self.subject)


@dataclass(frozen=True, slots=True)
class UnsupportedRelation:
    relation: str
    source_text: str


@dataclass(frozen=True, slots=True)
class AmbiguousRelation:
    """A known relation phrase lacking enough object roles for verification."""

    relation: str
    source_text: str
    reason: str = "requires_two_anchors"

@dataclass(frozen=True, slots=True)
class ProtectedEntityMention:
    placeholder: str
    text: str


@dataclass(frozen=True, slots=True)
class DecomposedEntity:
    text: str
    category: str
    mode: EntityMode
    source: str
    lexical_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParentheticalCue:
    """One parenthetical alias kept out of the primary dense description."""

    text: str
    language: Literal["vi", "en", "unknown"]


@dataclass(frozen=True, slots=True)
class SemanticView:
    """A complementary visual perspective encoded and retrieved independently."""

    role: SemanticViewRole
    text: str
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecomposedQuery:
    raw_query: str
    named_entities: tuple[DecomposedEntity, ...]
    visual_description_vi: str
    visual_description_en: str | None
    lexical_must_match: tuple[str, ...]
    semantic_views_vi: tuple[SemanticView, ...] = ()
    semantic_views_en: tuple[SemanticView, ...] = ()
    parenthetical_cues: tuple[ParentheticalCue, ...] = ()
    lexical_aliases: tuple[str, ...] = ()
    intent: QueryIntent = "visual_dense"
    spatial_constraints: tuple[SpatialConstraint, ...] = ()
    color_constraints: tuple[ColorConstraint, ...] = ()
    object_instances: tuple[ObjectInstance, ...] = ()
    sequence_constraints: tuple[SequenceConstraint, ...] = ()
    prospective_relations: tuple[ProspectiveRelation, ...] = ()
    object_nodes: tuple[ObjectNode, ...] = ()
    relative_constraints: tuple[RelativeSpatialConstraint, ...] = ()
    negation_constraints: tuple[NegationConstraint, ...] = ()
    unsupported_relations: tuple[UnsupportedRelation, ...] = ()
    ambiguous_relations: tuple[AmbiguousRelation, ...] = ()

    @property
    def required_entities(self) -> tuple[DecomposedEntity, ...]:
        return tuple(value for value in self.named_entities if value.mode == "required")

    @property
    def bound_constraint_count(self) -> int:
        return sum(
            value.bound
            for value in (
                *self.spatial_constraints,
                *self.color_constraints,
                *self.relative_constraints,
                *self.negation_constraints,
            )
        )

    @property
    def verification_constraint_count(self) -> int:
        return self.bound_constraint_count


def mask_entity_mentions(
    text: str,
    entities: Sequence[NamedEntity],
) -> tuple[str, tuple[ProtectedEntityMention, ...]]:
    """Replace every non-overlapping proper-name mention with a unique sentinel."""

    normalized = unicodedata.normalize("NFC", text)
    matches: list[tuple[int, int, str]] = []
    for entity in entities:
        terms = sorted({entity.text, *entity.aliases}, key=len, reverse=True)
        for term in terms:
            pattern = re.compile(r"(?<![\wÀ-ỹ])" + re.escape(term) + r"(?![\wÀ-ỹ])", re.IGNORECASE)
            for match in pattern.finditer(normalized):
                matches.append((match.start(), match.end(), match.group(0)))
    selected: list[tuple[int, int, str]] = []
    for value in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(value[0] < other[1] and other[0] < value[1] for other in selected):
            continue
        selected.append(value)
    selected.sort()
    if not selected:
        return normalized, ()
    parts: list[str] = []
    mentions: list[ProtectedEntityMention] = []
    cursor = 0
    for index, (start, end, value) in enumerate(selected):
        placeholder = f"__AIC_ENTITY_{index:03d}__"
        parts.extend((normalized[cursor:start], placeholder))
        mentions.append(ProtectedEntityMention(placeholder, value))
        cursor = end
    parts.append(normalized[cursor:])
    return "".join(parts), tuple(mentions)


def restore_entity_mentions(text: str, mentions: Sequence[ProtectedEntityMention]) -> str:
    restored = text
    for mention in mentions:
        if restored.count(mention.placeholder) != 1:
            raise ValueError("entity_placeholder_invalid")
        restored = restored.replace(mention.placeholder, mention.text, 1)
    return " ".join(unicodedata.normalize("NFC", restored).split())


def _hypernym(entity: NamedEntity, raw_query: str, *, english: bool) -> str:
    normalized = unicodedata.normalize("NFD", raw_query)
    folded = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d").replace("Đ", "D").casefold()
    if entity.category == "person":
        if "dinh than" in folded:
            return "a historical figure in a traditional Vietnamese temple" if english else "một nhân vật lịch sử trong đình thần truyền thống"
        return "a person" if english else "một người"
    if entity.category == "location":
        if any(token in folded for token in ("xa ", "nong thon", "mien tay")):
            return "a rural Vietnamese location" if english else "một địa điểm nông thôn Việt Nam"
        return "a Vietnamese location" if english else "một địa điểm ở Việt Nam"
    if any(token in folded for token in ("cau lac bo", "clb", "tu thien")):
        return "a charity club" if english else "một câu lạc bộ từ thiện"
    return "an organization" if english else "một tổ chức"


def _replace_entities(text: str, entities: Sequence[NamedEntity], raw_query: str, *, english: bool) -> str:
    value = text
    for entity in sorted(entities, key=lambda item: len(item.text), reverse=True):
        # The legacy public hint field also carries common concepts such as
        # foods. Treating every hint as an organisation corrupts the dense
        # prompt, so only server-extracted proper entities are hypernymized.
        if entity.source.startswith("operator_"):
            continue
        pattern = re.compile(r"(?<![\wÀ-ỹ])" + re.escape(entity.text) + r"(?![\wÀ-ỹ])", re.IGNORECASE)
        value = pattern.sub(_hypernym(entity, raw_query, english=english), value)
    value = re.sub(r"\s+", " ", value).strip(" ,.;")
    return value


def _tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", value, re.UNICODE)


def _clean_subject(tokens: Sequence[str], *, take_last: bool) -> str | None:
    values = [
        value
        for value in tokens
        if value.casefold() not in _SUBJECT_STOP and len(value) > 1
    ]
    if not values:
        return None
    selected = values[-4:] if take_last else values[:4]
    subject = " ".join(selected).strip()
    return subject or None


def _subject_around(text: str, start: int, end: int, *, prefer_before: bool) -> str | None:
    clause_start = max(text.rfind(separator, 0, start) for separator in (".", ";", ",", ":")) + 1
    following = [
        index for separator in (".", ";", ",", ":") if (index := text.find(separator, end)) >= 0
    ]
    clause_end = min(following) if following else len(text)
    before = _clean_subject(_tokens(text[clause_start:start]), take_last=True)
    after = _clean_subject(_tokens(text[end:clause_end]), take_last=False)
    return before or after if prefer_before else after or before


def _span_overlaps(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < blocked_end and blocked_start < end for blocked_start, blocked_end in spans)


def _color_in_text(text: str) -> str | None:
    matches: list[tuple[int, int, str]] = []
    for color, aliases in _COLOR_ALIASES.items():
        for alias in aliases:
            match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text, re.IGNORECASE)
            if match is not None:
                matches.append((match.start(), -(match.end() - match.start()), color))
    return min(matches)[2] if matches else None


def _without_color_words(text: str) -> str:
    value = text
    aliases = sorted(
        {alias for values in _COLOR_ALIASES.values() for alias in values},
        key=len,
        reverse=True,
    )
    for alias in aliases:
        value = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _parenthetical_parts(text: str) -> tuple[str, tuple[ParentheticalCue, ...]]:
    """Return primary text and normalized parenthetical lexical cues."""

    cues: list[ParentheticalCue] = []
    seen: set[str] = set()
    normalized = unicodedata.normalize("NFC", text)
    for match in _PARENTHETICAL.finditer(normalized):
        for raw in re.split(r"\s*(?:/|;|\bor\b|\bhoặc\b)\s*", match.group("text")):
            cue = " ".join(raw.split()).strip(" ,.;:-")
            identity = cue.casefold()
            if not cue or identity in seen:
                continue
            seen.add(identity)
            lowered = cue.casefold()
            language: Literal["vi", "en", "unknown"]
            if any(character in _VIETNAMESE_MARKED for character in lowered):
                language = "vi"
            elif all(ord(character) < 128 for character in cue):
                language = "en"
            else:
                language = "unknown"
            cues.append(ParentheticalCue(cue, language))
    primary = _PARENTHETICAL.sub(" ", normalized)
    primary = re.sub(r"\s+", " ", primary).strip(" ,.;")
    return primary, tuple(cues[:16])


def primary_query_text(text: str) -> str:
    """Return the dense-query text with parenthetical aliases removed."""

    return _parenthetical_parts(text)[0]


def sequence_constraints(text: str) -> tuple[SequenceConstraint, ...]:
    """Return bounded ordering idioms without assigning false event clauses."""

    normalized = unicodedata.normalize("NFC", text)
    matches: list[tuple[int, int, re.Match[str]]] = []
    for pattern in _SEQUENCE_PATTERNS:
        matches.extend((match.start(), match.end(), match) for match in pattern.finditer(normalized))
    output: list[SequenceConstraint] = []
    occupied: list[tuple[int, int]] = []
    for start, end, match in sorted(matches, key=lambda value: (value[0], -(value[1] - value[0]))):
        if _span_overlaps(start, end, occupied):
            continue
        occupied.append((start, end))
        label = match.groupdict().get("label")
        output.append(
            SequenceConstraint(
                "ordered_group",
                match.group(0),
                label.casefold() if label else None,
            )
        )
        if len(output) == 8:
            break
    return tuple(output)


def prospective_relations(text: str) -> tuple[ProspectiveRelation, ...]:
    """Return future-intent relations which are positive-only at verification time."""

    normalized = unicodedata.normalize("NFC", text)
    output: list[ProspectiveRelation] = []
    for match in _PROSPECTIVE_RELATION.finditer(normalized):
        anchor = " ".join(match.group("anchor").split()).strip(" ,.;:-")
        if not anchor:
            continue
        output.append(
            ProspectiveRelation(
                action=match.group("action").casefold(),
                anchor=anchor,
                source_text=match.group(0),
            )
        )
        if len(output) == 8:
            break
    return tuple(output)


def _coordinated_color_constraints(
    text: str,
    *,
    blocked_spans: Sequence[tuple[int, int]] = (),
) -> tuple[tuple[ColorConstraint, ...], tuple[tuple[int, int], ...]]:
    """Bind adjective and coordinated colors to the following noun phrase."""

    aliases = sorted(
        {alias for values in _COLOR_ALIASES.values() for alias in values},
        key=len,
        reverse=True,
    )
    alias_pattern = "(?:" + "|".join(re.escape(value) for value in aliases) + ")"
    separator = r"(?:\s+(?:and|và)\s+|\s*&\s*|\s*-\s*(?:and|và)\s*-\s*)"
    pattern = re.compile(
        rf"(?P<colors>{alias_pattern}(?:{separator}{alias_pattern})*)\s+"
        r"(?P<subject>[^,.;:]{1,80}?)"
        r"(?=\s+(?:and|và|with|filled|containing|garnished|on|in|wrapped|"
        r"wearing|without|no|left\s+of|right\s+of|above|below|inside|between|"
        r"prepare(?:s|d|ing)?|plan(?:s|ned|ning)?|wait(?:s|ed|ing)?|"
        r"cross(?:es|ed|ing)?|queue(?:s|d|ing)?|"
        r"cùng|với|chứa|trang\s+trí|được\s+bọc|không|bên\s+trái|bên\s+phải)"
        r"\b|[,.;:]|$)",
        re.IGNORECASE,
    )
    constraints: list[ColorConstraint] = []
    spans: list[tuple[int, int]] = []
    for match in pattern.finditer(text):
        if _span_overlaps(match.start(), match.end(), blocked_spans):
            continue
        # This pattern handles prenominal English adjectives. Vietnamese
        # postnominal colors are bound by _find_constraints instead.
        if any(character in _VIETNAMESE_MARKED for character in match.group("colors").casefold()):
            continue
        if re.match(r"(?:stands?|standing|walks?|walking|runs?|running)\b",
                    match.group("subject"), re.IGNORECASE):
            continue
        subject = _clean_subject(_tokens(match.group("subject")), take_last=False)
        subject = _without_color_words(subject) if subject else None
        if not subject:
            continue
        colors: list[str] = []
        for color, color_aliases in _COLOR_ALIASES.items():
            if any(
                re.search(
                    r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                    match.group("colors"),
                    re.IGNORECASE,
                )
                for alias in color_aliases
            ):
                colors.append(color)
        if not colors:
            continue
        spans.append((match.start(), match.end()))
        constraints.extend(
            ColorConstraint(subject, color, match.group(0)) for color in colors
        )
    return tuple(constraints[:16]), tuple(spans[:16])


_INSTANCE_CANONICAL_LABELS: tuple[tuple[str, frozenset[str]], ...] = (
    ("vehicle", frozenset({"car", "cars", "vehicle", "vehicles", "automobile", "ô tô", "xe hơi"})),
    ("person", frozenset({"person", "people", "man", "woman", "người"})),
)


def _canonical_instance_label(subject: str) -> str:
    folded = fold_entity_text(subject)
    tokens = set(folded.split())
    for canonical, aliases in _INSTANCE_CANONICAL_LABELS:
        if folded in aliases or tokens & aliases:
            return canonical
    return folded or subject.casefold()


def _bind_color_instances(
    constraints: Sequence[ColorConstraint],
) -> tuple[tuple[ColorConstraint, ...], tuple[ObjectInstance, ...]]:
    """Bind repeated attribute-object mentions to stable, one-to-one instance IDs."""

    grouped: dict[tuple[str, str, str], list[tuple[int, ColorConstraint]]] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    for index, constraint in enumerate(constraints):
        subject = constraint.subject.strip() if constraint.subject else ""
        if not subject:
            continue
        subject_identity = fold_entity_text(subject)
        source_identity = " ".join(constraint.source_text.casefold().split())
        existing_node = constraint.node_id or ""
        key = (subject_identity, source_identity, existing_node)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append((index, constraint))

    per_label_ordinal: dict[str, int] = {}
    node_by_index: dict[int, str] = {}
    instances: list[ObjectInstance] = []
    for subject_identity, source_identity, existing_node in ordered_keys:
        values = grouped[(subject_identity, source_identity, existing_node)]
        display = values[0][1].subject or subject_identity
        canonical = _canonical_instance_label(display)
        ordinal = per_label_ordinal.get(canonical, 0) + 1
        per_label_ordinal[canonical] = ordinal
        safe_label = re.sub(r"[^a-z0-9]+", "_", canonical).strip("_") or "object"
        instance_id = existing_node or f"{safe_label}_{ordinal}"
        group_id = f"{safe_label}_group_1"
        attributes = tuple(dict.fromkeys(value.color for _index, value in values))
        instances.append(
            ObjectInstance(
                instance_id=instance_id,
                label=display,
                canonical_label=canonical,
                attributes=attributes,
                group_id=group_id,
                ordinal=ordinal,
                source_text=values[0][1].source_text,
            )
        )
        for index, _constraint in values:
            node_by_index[index] = instance_id

    rebound = tuple(
        ColorConstraint(
            subject=value.subject,
            color=value.color,
            source_text=value.source_text,
            node_id=node_by_index.get(index, value.node_id),
        )
        for index, value in enumerate(constraints)
    )
    return rebound, tuple(instances[:16])


def _semantic_views(
    visual_text: str | None,
    colors: Sequence[ColorConstraint],
    object_nodes: Sequence[ObjectNode],
    object_instances: Sequence[ObjectInstance] = (),
    sequences: Sequence[SequenceConstraint] = (),
    prospective: Sequence[ProspectiveRelation] = (),
) -> tuple[SemanticView, ...]:
    """Build bounded structural views without averaging away rare details."""

    if visual_text is None:
        return ()
    global_text = re.sub(r"\s+", " ", visual_text).strip(" ,.;")
    if not global_text:
        return ()
    candidates: list[SemanticView] = [SemanticView("global", global_text)]
    # Preserve the historical single-encode path for concise queries. Multiple
    # views are useful only once attributes/clauses are numerous enough to risk
    # semantic dilution; a simple "person in red" must remain one search.
    is_detail_rich = bool(
        len(_tokens(global_text)) >= 12
        or re.search(r"[,.;:]", global_text)
        or len(colors) >= 3
        or len(object_nodes) >= 3
    )
    if not is_detail_rich:
        return tuple(candidates)
    first_clause = re.split(r"[,.;:]", global_text, maxsplit=1)[0]
    action = _ACTION_BOUNDARY.split(first_clause, maxsplit=1)[0].strip(" ,.;")
    has_structural_detail = bool(colors or object_nodes or re.search(r"[,.;:]", global_text))
    structured_sequence = bool(sequences or prospective)
    if has_structural_detail and len(_tokens(action)) >= 2:
        candidates.append(
            SemanticView("scene_action" if structured_sequence else "action", action)
        )
    if not structured_sequence:
        subjects: list[str] = []
        subject_identities: set[str] = set()
        for value in (*colors, *object_nodes):
            subject = value.subject if isinstance(value, ColorConstraint) else value.label
            if subject and subject.casefold() not in subject_identities:
                subjects.append(subject)
                subject_identities.add(subject.casefold())
        if subjects:
            candidates.append(SemanticView("foreground_subject", "; ".join(subjects)))

    grouped: dict[str, list[str]] = {}
    displays: dict[str, str] = {}
    for constraint in colors:
        if not constraint.bound or constraint.subject is None:
            continue
        identity = fold_entity_text(constraint.subject)
        displays.setdefault(identity, constraint.subject)
        grouped.setdefault(identity, [])
        if constraint.color not in grouped[identity]:
            grouped[identity].append(constraint.color)
    repeated_instance_labels = {
        value.canonical_label
        for value in object_instances
        if sum(
            other.canonical_label == value.canonical_label
            for other in object_instances
        ) > 1
    }
    if repeated_instance_labels:
        attributes = [
            f"{' and '.join(value.attributes)} {value.canonical_label}"
            for value in object_instances
            if value.canonical_label in repeated_instance_labels and value.attributes
        ]
    else:
        attributes = [
            f"{' and '.join(grouped[identity])} {displays[identity]}"
            for identity in grouped
        ]
    if attributes:
        candidates.append(
            SemanticView(
                "instance_attributes"
                if structured_sequence and repeated_instance_labels
                else "distinctive_attributes",
                "; ".join(attributes),
            )
        )

    if prospective and repeated_instance_labels:
        repeated = next(
            value for value in object_instances
            if value.canonical_label in repeated_instance_labels
        )
        instance_values = [
            value
            for value in object_instances
            if value.canonical_label == repeated.canonical_label and value.attributes
        ]
        colors_in_order = [value.attributes[0] for value in instance_values]
        if colors_in_order:
            color_text = (
                colors_in_order[0]
                if len(colors_in_order) == 1
                else f"{', '.join(colors_in_order[:-1])} and {colors_in_order[-1]}"
            )
            plural = (
                "people"
                if repeated.canonical_label == "person"
                else f"{repeated.canonical_label}s"
            )
            lowered = global_text.casefold()
            environment = "in water" if "water" in lowered else ""
            anchor = prospective[0].anchor
            anchor = re.sub(r"^(?:a|an|the)\s+", "", anchor, flags=re.IGNORECASE)
            primary = " ".join(
                value for value in (
                    color_text,
                    plural,
                    environment,
                    f"at a {anchor}" if anchor else "",
                ) if value
            )
            alternatives = tuple(
                " ".join(
                    value for value in (
                        color_text,
                        plural,
                        environment,
                        f"{preposition} a {anchor}" if anchor else "",
                    ) if value
                )
                for preposition in ("by", "near")
            )
            candidates.append(
                SemanticView("relation_relaxed", primary, alternatives)
            )

    output: list[SemanticView] = []
    seen: set[str] = set()
    for candidate in candidates:
        identity = " ".join(candidate.text.casefold().split())
        if identity in seen:
            continue
        seen.add(identity)
        output.append(candidate)
        if len(output) == 4:
            break
    return tuple(output)


_ROLE_BOUNDARY = re.compile(
    r"\b(?:and|but|while|then|with|without|no|và|nhưng|rồi|sau\s+đó|cùng\s+với|không|"
    r"left\s+of|right\s+of|above|below|inside|within|contains|between|near|"
    r"far\s+from|overlapping|surrounding|in\s+front\s+of|behind|"
    r"bên\s+trái|bên\s+phải|phía\s+trên|phía\s+dưới|bên\s+trong|giữa|"
    r"gần|cách\s+xa|bao\s+quanh|chồng\s+lên|phía\s+trước|phía\s+sau)\b",
    re.IGNORECASE,
)


def _bounded_role_fragment(text: str) -> str:
    match = _ROLE_BOUNDARY.search(text)
    return text[: match.start()] if match is not None else text


def _relation_role(text: str, *, take_last: bool) -> tuple[str | None, str | None]:
    cleaned = re.sub(r"^(?:and|và)\s+", "", text, flags=re.IGNORECASE)
    fragment = _bounded_role_fragment(cleaned)
    color = _color_in_text(fragment)
    without_color = _without_color_words(fragment)
    without_color = re.split(
        r"\b(?:stands?|standing|walks?|walking|runs?|running|đứng|đi|chạy)\b",
        without_color, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    label = _clean_subject(_tokens(without_color), take_last=take_last)
    return label, color


def _find_negations(
    raw_query: str,
) -> tuple[tuple[NegationConstraint, ...], tuple[tuple[int, int], ...]]:
    normalized = unicodedata.normalize("NFC", raw_query)
    constraints: list[NegationConstraint] = []
    spans: list[tuple[int, int]] = []
    for match in _NEGATION_PATTERN.finditer(normalized):
        raw_subject = _bounded_role_fragment(match.group("subject")).strip()
        if not raw_subject:
            continue
        subject, color = _relation_role(raw_subject, take_last=False)
        subject_end = match.start("subject") + len(raw_subject)
        spans.append((match.start(), subject_end))
        constraints.append(
            NegationConstraint(
                subject=subject,
                source_text=normalized[match.start():subject_end],
                color=color,
            )
        )
    return tuple(constraints[:16]), tuple(spans[:16])


def _strip_negated_clauses(text: str) -> str:
    _constraints, spans = _find_negations(text)
    if not spans:
        return re.sub(r"\s+", " ", text).strip(" ,.;")
    parts: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start >= cursor:
            parts.append(text[cursor:start])
            cursor = end
    parts.append(text[cursor:])
    return re.sub(r"\s+", " ", "".join(parts)).strip(" ,.;")


def _find_relations(
    raw_query: str,
    *,
    blocked_spans: Sequence[tuple[int, int]] = (),
) -> tuple[
    tuple[ObjectNode, ...],
    tuple[RelativeSpatialConstraint, ...],
    tuple[ColorConstraint, ...],
    tuple[UnsupportedRelation, ...],
    tuple[AmbiguousRelation, ...],
    tuple[tuple[int, int], ...],
]:
    normalized = unicodedata.normalize("NFC", raw_query)
    raw_matches: list[
        tuple[int, int, SpatialRelation, re.Match[str], int]
    ] = []
    for relation, pattern in _TERNARY_RELATION_PATTERNS:
        for match in pattern.finditer(normalized):
            if not any(start <= match.start() < end for start, end in blocked_spans):
                raw_matches.append((match.start(), match.end(), relation, match, 3))
    for relation, pattern in _BINARY_RELATION_PATTERNS:
        for match in pattern.finditer(normalized):
            if not any(start <= match.start() < end for start, end in blocked_spans):
                raw_matches.append((match.start(), match.end(), relation, match, 2))

    selected: list[tuple[int, int, SpatialRelation, re.Match[str], int]] = []
    for value in sorted(raw_matches, key=lambda item: (item[0], -(item[1] - item[0]), -item[4])):
        if _span_overlaps(value[0], value[1], [(item[0], item[1]) for item in selected]):
            continue
        selected.append(value)

    nodes: list[ObjectNode] = []
    relations: list[RelativeSpatialConstraint] = []
    colors: list[ColorConstraint] = []
    for index, (_start, _end, relation, match, arity) in enumerate(selected[:16]):
        if arity == 3:
            subject, subject_color = _relation_role(match.group("b"), take_last=True)
            object_subject, object_color = _relation_role(match.group("a"), take_last=False)
            second_subject, second_color = _relation_role(match.group("c"), take_last=False)
        else:
            subject, subject_color = _relation_role(match.group("a"), take_last=True)
            object_subject, object_color = _relation_role(match.group("b"), take_last=False)
            second_subject = second_color = None
        subject_node = f"relation_{index}_subject"
        object_node = f"relation_{index}_object"
        second_node = f"relation_{index}_second" if arity == 3 else None
        if subject:
            nodes.append(ObjectNode(subject_node, subject, match.group("b" if arity == 3 else "a").strip()))
        if object_subject:
            nodes.append(ObjectNode(object_node, object_subject, match.group("a" if arity == 3 else "b").strip()))
        if second_subject and second_node:
            nodes.append(ObjectNode(second_node, second_subject, match.group("c").strip()))
        relations.append(
            RelativeSpatialConstraint(
                subject=subject,
                relation=relation,
                object_subject=object_subject,
                source_text=match.group(0),
                subject_node_id=subject_node,
                object_node_id=object_node,
                second_object_subject=second_subject,
                second_object_node_id=second_node,
            )
        )
        for role_subject, role_color, role_text, node_id in (
            (subject, subject_color, match.group("b" if arity == 3 else "a"), subject_node),
            (object_subject, object_color, match.group("a" if arity == 3 else "b"), object_node),
            (second_subject, second_color, match.group("c") if arity == 3 else "", second_node),
        ):
            if role_subject and role_color and node_id:
                colors.append(ColorConstraint(role_subject, role_color, role_text.strip(), node_id))

    relation_spans = tuple((start, end) for start, end, _relation, _match, _arity in selected)
    ambiguous: list[AmbiguousRelation] = []
    for match in _AMBIGUOUS_BETWEEN.finditer(normalized):
        if _span_overlaps(match.start(), match.end(), (*blocked_spans, *relation_spans)):
            continue
        ambiguous.append(AmbiguousRelation("between", match.group(0)))
    return (
        tuple(nodes),
        tuple(relations),
        tuple(colors),
        (),
        tuple(ambiguous[:16]),
        relation_spans,
    )

def _find_constraints(
    raw_query: str,
    *,
    blocked_spans: Sequence[tuple[int, int]] = (),
) -> tuple[tuple[SpatialConstraint, ...], tuple[ColorConstraint, ...]]:
    normalized = unicodedata.normalize("NFC", raw_query)
    spatial: list[SpatialConstraint] = []
    coordinated_colors, coordinated_spans = _coordinated_color_constraints(
        normalized,
        blocked_spans=blocked_spans,
    )
    colors: list[ColorConstraint] = list(coordinated_colors)
    occupied_spatial: list[tuple[str, int, int]] = []
    occupied_colors: list[tuple[str, int, int]] = []

    for zone, aliases in _ZONE_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(normalized):
                if _span_overlaps(
                    match.start(),
                    match.end(),
                    (*blocked_spans, *coordinated_spans),
                ):
                    continue
                identity = (zone, match.start(), match.end())
                if any(
                    previous_zone == zone
                    and match.start() < previous_end
                    and previous_start < match.end()
                    for previous_zone, previous_start, previous_end in occupied_spatial
                ):
                    continue
                occupied_spatial.append(identity)
                subject = _subject_around(
                    normalized,
                    match.start(),
                    match.end(),
                    prefer_before=not alias.startswith(("ở ", "bên ", "phía ")),
                )
                spatial.append(
                    SpatialConstraint(
                        subject=subject,
                        zone=zone,  # type: ignore[arg-type]
                        source_text=match.group(0),
                        perspective_proxy=alias.casefold()
                        in {"foreground", "background", "tiền cảnh", "hậu cảnh"},
                    )
                )

    for color, aliases in _COLOR_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(normalized):
                if _span_overlaps(
                    match.start(),
                    match.end(),
                    (*blocked_spans, *coordinated_spans),
                ):
                    continue
                identity = (color, match.start(), match.end())
                if any(
                    previous_color == color
                    and match.start() < previous_end
                    and previous_start < match.end()
                    for previous_color, previous_start, previous_end in occupied_colors
                ):
                    continue
                occupied_colors.append(identity)
                before_text = normalized[max(0, match.start() - 12):match.start()].casefold()
                # Vietnamese color adjectives follow the noun. English usually
                # puts them first, except predicates such as "person in red".
                vietnamese_color = alias not in {
                    "red", "blue", "yellow", "green", "black", "white", "pink",
                    "purple", "violet",
                }
                predicate_color = bool(re.search(
                    r"\b(?:in|is|are|wearing)\s*$", before_text
                ))
                prefer_before = (
                    vietnamese_color or predicate_color or "màu" in before_text
                    or "color" in before_text
                )
                subject = _subject_around(
                    normalized,
                    match.start(),
                    match.end(),
                    prefer_before=prefer_before,
                )
                if subject and vietnamese_color and "mặc" in _tokens(subject.casefold()):
                    subject = re.split(r"\bmặc\b", subject, flags=re.IGNORECASE)[-1].strip()
                colors.append(
                    ColorConstraint(
                        _without_color_words(subject) if subject else None,
                        color,
                        match.group(0),
                    )
                )
    return tuple(spatial), tuple(colors)


def _intent(
    raw_query: str,
    entities: Sequence[DecomposedEntity],
    spatial: Sequence[SpatialConstraint],
    colors: Sequence[ColorConstraint],
    relatives: Sequence[RelativeSpatialConstraint],
    negations: Sequence[NegationConstraint],
) -> QueryIntent:
    if any(value.bound for value in (*spatial, *colors, *relatives, *negations)):
        return "spatial_color"
    if entities or _QUOTED.search(raw_query) or _NUMBER.search(raw_query) or _LEXICAL_CUES.search(raw_query):
        return "lexical_dominant"
    return "visual_dense"



def decompose_query(
    raw_query: str,
    entities_with_modes: Sequence[tuple[NamedEntity, EntityMode]],
    *,
    english_text: str | None,
    manual_english_override: bool = False,
) -> DecomposedQuery:
    entities = tuple(entity for entity, _mode in entities_with_modes)
    decomposed_entities = tuple(
        DecomposedEntity(
            entity.text,
            entity.category,
            mode,
            entity.source,
            entity.lexical_terms(),
        )
        for entity, mode in entities_with_modes
    )
    primary_raw_query, parenthetical_cues = _parenthetical_parts(raw_query)
    terms: list[str] = []
    seen: set[str] = set()
    for value in decomposed_entities:
        for term in value.lexical_terms:
            identity = term.casefold()
            if identity not in seen:
                terms.append(term)
                seen.add(identity)
    lexical_aliases: list[str] = []
    alias_seen: set[str] = set()
    for value in (
        *(cue.text for cue in parenthetical_cues),
        *(
            entity.text
            for entity in decomposed_entities
            if entity.source.startswith("operator_")
        ),
    ):
        identity = value.casefold()
        if identity not in alias_seen:
            lexical_aliases.append(value)
            alias_seen.add(identity)
    visual_en = english_text
    if visual_en is not None and not manual_english_override:
        primary_en, _english_cues = _parenthetical_parts(visual_en)
        visual_en = _replace_entities(primary_en, entities, raw_query, english=True)

    negation_constraints, negated_spans = _find_negations(raw_query)
    prospective = prospective_relations(raw_query)
    sequences = tuple(
        SequenceConstraint(
            value.kind,
            value.source_text,
            value.member_label,
            prospective[0].action if prospective else None,
            value.confidence,
        )
        for value in sequence_constraints(raw_query)
    )
    (
        relation_nodes,
        relative_constraints,
        relation_colors,
        unsupported_relations,
        ambiguous_relations,
        relation_spans,
    ) = _find_relations(raw_query, blocked_spans=negated_spans)
    spatial_constraints, raw_color_constraints = _find_constraints(
        raw_query,
        blocked_spans=(*negated_spans, *relation_spans),
    )
    colors: list[ColorConstraint] = []
    color_keys: set[tuple[str, str, str, str]] = set()
    for constraint in (*relation_colors, *raw_color_constraints):
        subject = constraint.subject.strip() if constraint.subject else ""
        subject_identity = fold_entity_text(subject) if subject else ""
        key = (
            subject_identity,
            constraint.color,
            " ".join(constraint.source_text.casefold().split()),
            constraint.node_id or "",
        )
        if key in color_keys:
            continue
        color_keys.add(key)
        colors.append(
            ColorConstraint(
                subject=subject or None,
                color=constraint.color,
                source_text=constraint.source_text,
                node_id=constraint.node_id,
            )
        )
    color_constraints, object_instances = _bind_color_instances(colors[:16])

    object_nodes: list[ObjectNode] = list(relation_nodes)
    seen_nodes = {node.node_id for node in object_nodes}
    instance_by_node = {
        value.instance_id: value for value in object_instances
    }
    for prefix, constraints in (
        ("absolute", spatial_constraints),
        ("color", color_constraints),
        ("negative", negation_constraints),
    ):
        for index, constraint in enumerate(constraints):
            subject = constraint.subject.strip() if constraint.subject else ""
            if not subject:
                continue
            node_id = getattr(constraint, "node_id", None) or f"{prefix}_{index}"
            if node_id not in seen_nodes:
                instance = instance_by_node.get(node_id)
                object_nodes.append(
                    ObjectNode(
                        node_id,
                        instance.canonical_label if instance is not None else subject,
                        constraint.source_text,
                    )
                )
                seen_nodes.add(node_id)

    visual_vi = _replace_entities(primary_raw_query, entities, raw_query, english=False)
    visual_vi = _strip_negated_clauses(visual_vi)
    if visual_en is not None:
        visual_en = _strip_negated_clauses(visual_en)
    semantic_views_vi = _semantic_views(
        visual_vi,
        color_constraints,
        object_nodes,
        object_instances,
        sequences,
        prospective,
    )
    semantic_views_en = _semantic_views(
        visual_en,
        color_constraints,
        object_nodes,
        object_instances,
        sequences,
        prospective,
    )
    return DecomposedQuery(
        raw_query=raw_query,
        named_entities=decomposed_entities,
        visual_description_vi=visual_vi,
        visual_description_en=visual_en,
        lexical_must_match=tuple(terms),
        semantic_views_vi=semantic_views_vi,
        semantic_views_en=semantic_views_en,
        parenthetical_cues=parenthetical_cues,
        lexical_aliases=tuple(lexical_aliases),
        intent=_intent(
            raw_query,
            decomposed_entities,
            spatial_constraints,
            color_constraints,
            relative_constraints,
            negation_constraints,
        ),
        spatial_constraints=spatial_constraints,
        color_constraints=color_constraints,
        object_instances=object_instances,
        sequence_constraints=sequences,
        prospective_relations=prospective,
        object_nodes=tuple(object_nodes[:32]),
        relative_constraints=relative_constraints,
        negation_constraints=negation_constraints,
        unsupported_relations=unsupported_relations,
        ambiguous_relations=ambiguous_relations,
    )
