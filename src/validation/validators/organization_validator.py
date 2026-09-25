"""Hard-rejects candidates containing a known organization/institution
keyword ('Little Flowers High School', 'Sri Lakshmi Ganesh Traders',
'Rajupalem Municipal Office'). Keyword-based (not exact-phrase) so it
generalizes to NOVEL business/institution names, the same rationale as
CampaignValidator's marker words."""

from __future__ import annotations

from src.core.models import CandidateResult, ValidationResult
from src.knowledge.knowledge_base import KnowledgeBase
from src.validation.base_validator import BaseValidator

ORG_KEYWORDS = frozenset({
    "school", "college", "university", "institute", "academy",
    "hospital", "clinic", "pharmacy",
    "traders", "enterprises", "industries", "stores", "store", "mart",
    "company", "corporation", "corp", "ltd", "limited", "inc", "llp",
    "bank", "trust", "foundation",
    "council", "committee", "department", "ministry", "party",
    "union", "association", "society", "board", "authority", "agency",
    "bureau", "commission",
    "municipal", "municipality",
    "office", "hall", "centre", "center", "complex", "plaza", "mall",
    "hotel", "restaurant", "services", "solutions", "group",
    "alliance",  # found via testing (case 930204710): a fictional charity
    # ("Coastal Relief Alliance") had no other keyword in this list and
    # slipped through into REVIEW as a 3-word "person" - "alliance" is
    # exactly the same class of generic institutional word already
    # covered here (society/union/association/foundation/trust).
    "division",  # found via real casework (2026-09-16, real case
    # case scan): "Windows Division" cleared into REVIEW with no keyword
    # here to catch it (only the decision engine's separate common-word-
    # phrase guard stopped it from reaching ACCEPTED) - "division" is the
    # same class of generic business-unit word as "department"/"board"
    # already covered here, so it generalizes the same way for any other
    # "<Product/Brand> Division" naming pattern, not just this one case.
    "licensing", "marketplace", "warranty", "software",  # found via real
    # casework (2026-09-17, real case scan, a EULA/license
    # document): "Select Licensing", "Provider Licensing", "Zune
    # Marketplace", "Product Warranty", "MontaVista Software" all cleared
    # into REVIEW with no keyword here to catch any of them - same class
    # of generic product/business-suffix word as "company"/"solutions"
    # already covered above, so this generalizes to any other
    # "<Product/Brand> Licensing/Marketplace/Warranty/Software" pattern
    # in similar EULA/license-style documents, not just these 5 cases.
    "consultancy", "consulting", "consultants", "holdings", "ventures",
    "technologies", "logistics",  # added 2026-09-24: a contacts-CSV
    # company column ("Tata Consultancy") was ACCEPTED as a person - "Tata"
    # is a real surname and nothing here marked "Consultancy" as a business
    # word. Same generic company-suffix class as "services"/"solutions";
    # checked against every benchmark gold name first (no collisions).
    "customercare", "helpline", "helpdesk", "support",  # added 2026-09-24:
    # a real case scan ACCEPTED "Bsnl Customercare", "Jio Customercare",
    # "Airtel Customercare", "Sony Customercare" (phone contact labels) -
    # support-line names, the same class as "services". No gold collisions.
})


class OrganizationValidator(BaseValidator):
    name = "organization"

    def validate(self, candidate: CandidateResult, knowledge_base: KnowledgeBase, document_text: str) -> ValidationResult:
        tokens = candidate.candidate.normalized_text.split()
        org_tokens = [t for t in tokens if t.lower().rstrip(".,") in ORG_KEYWORDS]

        if org_tokens:
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"Contains organization/institution keyword: {', '.join(org_tokens)}",
            )

        if knowledge_base.is_organization(candidate.candidate.normalized_text):
            return ValidationResult(
                self.name, passed=False, severity="hard",
                message=f"'{candidate.candidate.normalized_text}' matches known organization",
            )

        return ValidationResult(self.name, passed=True, score=0.0, message="No organization keywords")
