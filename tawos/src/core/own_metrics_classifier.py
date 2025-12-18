import re
import time
from core.base_classifier import BaseClassifier
from core.log import get_logger
from config_loader import config


def _calculate_length_score(title: str, description: str) -> int:  # 0-30 points
    score = 0

    if len(title) >= 10:
        score += 5
    if len(title) >= 20:
        score += 3
    if len(title) >= 30:
        score += 2
    if len(title) >= 40:
        score += 2

    if len(description) >= 50:
        score += 5
    if len(description) >= 100:
        score += 5
    if len(description) >= 200:
        score += 5
    if len(description) >= 500:
        score += 5

    return min(score, 30)


def _calculate_structure_score(description: str) -> int:  # 0-10 points
    score = 0

    if "\n" in description:
        score += 3
    if description.count("\n") >= 3:
        score += 3
    if re.search(r"(- |• |\* )", description):
        score += 4

    return min(score, 10)


def _calculate_style_score(title: str, description: str) -> int:  # 0-20 points
    score = 20

    text = title + " " + description

    special_char_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?;:()\-]", text)) / max(
        len(text), 1
    )
    if special_char_ratio > 0.3:
        score -= 10
    elif special_char_ratio > 0.15:
        score -= 5

    if re.search(r"[!?\.]{3,}", text):
        score -= 8

    caps_ratio = len(re.findall(r"[A-Z]", text)) / max(
        len(re.findall(r"[a-zA-Z]", text)), 1
    )
    if caps_ratio > 0.5 and len(text) > 20:
        score -= 10

    return max(score, 0)


def _calculate_professionalism_score(
    title: str, description: str
) -> int:  # 0-20 points
    score = 20

    text = (title + " " + description).lower()

    spam_severe = [
        r"\bfree\b",
        r"\bwin\b",
        r"\boffer\b",
        r"\bclick here\b",
        r"\bvirus\b",
        r"\bhack\b",
        r"\bcheat\b",
    ]
    for pattern in spam_severe:
        if re.search(pattern, text):
            score -= 10
            break

    unprofessional = [
        r"\bxd+\b",
        r"\blol+\b",
        r"\blomg\b",
        r"\bwtf\b",
        r"\blmao\b",
        r"\byo\b",
        r"\bhey+\b",
        r"\bpls\b",
        r"\bthx\b",
        r"\bu\b",
    ]
    for pattern in unprofessional:
        if re.search(pattern, text):
            score -= 2

    aggressive = [
        r"\bstupid\b",
        r"\bdumb\b",
        r"\bidiot\b",
        r"\bmoron\b",
        r"\bimbecile\b",
        r"\bfool\b",
        r"\bloser\b",
        r"\bpathetic\b",
        r"\bworthless\b",
        r"\buseless\b",
        r"\bincompetent\b",
        r"\bclueless\b",
        r"\bignorant\b",
        r"\bcrap\b",
        r"\bsucks\b",
        r"\bdamn\b",
        r"\bhell\b",
        r"\bass\b",
        r"\bbastard\b",
        r"\bbitch\b",
        r"\bfuck",
        r"\bshit",
        r"\bpiss",
        r"\bshut up\b",
        r"\bwhatever\b",
        r"\bdon't care\b",
        r"\bwho cares\b",
        r"\bbig deal\b",
        r"\bso what\b",
        r"\bgarbage\b",
        r"\btrash\b",
        r"\brubbish\b",
        r"\bjunk\b",
        r"\bwaste of (time|space)\b",
        r"\bthreat\b",
        r"\bwarn(ing)?\b.*\byou\b",
        r"\bregret\b.*\bthis\b",
        r"\bpay for this\b",
        r"\bget you\b",
        r"\bpathetic\b",
        r"\bjoke\b",
        r"\blaughing at you\b",
        r"\bridiculous\b",
        r"\babsurd\b",
        r"\blame\b",
        r"\bweak\b",
        r"\bcoward\b",
        r"\blogic\b.*\bfail\b",
        r"\bangry\b",
        r"\bfurious\b",
        r"\benraged\b",
        r"\bpissed off\b",
        r"\bsick of (you|this)\b",
        r"\bhad enough\b",
        r"\blast straw\b",
        r"\bso-called\b",
        r"\bsupp(osedly|osed)\b",
        r"\ballegedly\b.*\b(expert|professional)\b",
        r"\bself-proclaimed\b",
        r"\bwannabe\b",
        r"\bliar\b",
        r"\bcheat(er)?\b",
        r"\bfraud\b",
        r"\bfake\b",
        r"\bdishonest\b",
        r"\buntrustworthy\b",
        r"\bscammer\b",
        r"\bget out\b",
        r"\bdon't belong\b",
        r"\bnot welcome\b",
        r"\bnobody likes you\b",
        r"\beveryone hates you\b",
    ]
    for pattern in aggressive:
        if re.search(pattern, text):
            score -= 5
            break

    placeholder_patterns = [
        r"\btest\s*\d*$",
        r"\basdf+\b",
        r"\bqwerty\b",
        r"\btodo\s*:?\s*$",
        r"\bfixme\s*:?\s*$",
        r"\bplaceholder\b",
    ]
    for pattern in placeholder_patterns:
        if re.search(pattern, text):
            score -= 5

    return max(score, 0)


def _calculate_technical_content_score(
    title: str, description: str
) -> int:  # 0-20 points
    score = 0

    text = title + " " + description

    technical_indicators = [
        r"\b(bug|error|exception|crash|fail|failure|broken|issue|defect|glitch)\b",
        r"\b(API|endpoint|database|query|SQL|NoSQL|server|backend|microservice|service)\b",
        r"\b(Docker|container|Kubernetes|K8s|deployment|infrastructure|cloud|AWS|Azure|GCP)\b",
        r"\b(function|method|class|module|component|library|package|framework)\b",
        r"\b(variable|constant|parameter|argument|return|callback|promise|async|await)\b",
        r"\b(user|customer|client|interface|UI|UX|frontend|design|layout|responsive)\b",
        r"\b(performance|optimization|memory|speed|latency|throughput|bottleneck|cache|caching)\b",
        r"\b(load time|bandwidth|scalability|concurrent|parallel|threading)\b",
        r"\b(security|authentication|authorization|permission|encryption|vulnerability|exploit)\b",
        r"\b(token|JWT|OAuth|SSL|TLS|certificate|firewall|injection|XSS|CSRF)\b",
        r"\b(test|testing|QA|validation|unit test|integration|E2E|coverage|assertion)\b",
        r"\b(debug|debugging|breakpoint|logging|monitoring|trace|stack trace)\b",
        r"\b(feature|enhancement|improvement|request|requirement|specification|story)\b",
        r"\b(implementation|develop|build|create|integrate|configure)\b",
        r"\b(refactor|cleanup|technical debt|dependency|deprecate|legacy|migrate|migration)\b",
        r"\b(version|upgrade|update|patch|release|hotfix|rollback)\b",
        r"\b(architecture|design pattern|MVC|REST|GraphQL|WebSocket|RPC|SOAP)\b",
        r"\b(schema|model|entity|DTO|serialization|middleware|pipeline)\b",
        r"\b(git|commit|branch|merge|pull request|PR|pipeline|CI/CD|build|compile)\b",
        r"\b(data|dataset|JSON|XML|CSV|serialize|parse|store|retrieve|CRUD)\b",
        r"\b(backup|restore|sync|replication|index|transaction|consistency)\b",
        r"\b(HTTP|HTTPS|request|response|status code|timeout|retry|webhook)\b",
        r"\b(socket|stream|protocol|header|payload|connection|session)\b",
        r"\b(algorithm|loop|iteration|recursion|condition|logic|syntax|compile)\b",
        r"\b(runtime|execution|process|thread|queue|stack|heap|pointer)\b",
    ]
    matches = 0
    for pattern in technical_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1

    if matches >= 1:
        score += 5
    if matches >= 2:
        score += 5
    if matches >= 3:
        score += 5
    if matches >= 4:
        score += 5

    if re.search(r"```|`[^`]+`|\n\s{4,}", description):
        score += 5

    if re.search(r"https?://|/[a-z/]+\.[a-z]+|v?\d+\.\d+", text):
        score += 3

    return min(score, 20)


def _detect_spam_or_invalid(title: str, description: str) -> bool:
    text = (title + " " + description).lower()

    if len(title.strip()) < 5 or len(description.strip()) < 10:
        return True

    spam_patterns = [
        r"click here.*free",
        r"winner.*prize",
        r"congratulations.*won",
        r"buy now.*discount",
    ]

    for pattern in spam_patterns:
        if re.search(pattern, text):
            return True

    return False


class OwnMetricsClassifier(BaseClassifier):
    def __init__(self) -> None:
        super().__init__(
            model_name="OwnMetrics",
            batch_size=config.LOCAL_MODEL_BATCH_SIZE,
            timing_data=[],
        )
        self.logger = get_logger(self.model_name)

    async def _get_model_name(self) -> str:
        return "OwnMetrics"

    async def _classify_single(
        self, session: object, title: str, desc: str
    ) -> int | None:
        title = title if isinstance(title, str) else ""
        desc = desc if isinstance(desc, str) else ""

        title = title.replace('"', "")
        desc = desc.replace('"', "")

        start_time = time.time()

        if _detect_spam_or_invalid(title, desc):
            score = 0
        else:
            length = _calculate_length_score(title, desc)
            structure = _calculate_structure_score(desc)
            style = _calculate_style_score(title, desc)
            professionalism = _calculate_professionalism_score(title, desc)
            technical = _calculate_technical_content_score(title, desc)

            score = length + structure + style + professionalism + technical
            score = max(0, min(100, score))

        elapsed_time = time.time() - start_time
        self.timing_data.append(elapsed_time)

        return score
