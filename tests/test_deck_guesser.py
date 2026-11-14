import pytest

from backend_engine.perception.bayesian_deck_guesser import BayesianDeckGuesser, default_guesser, known_costs

HOG26 = ["hog_rider", "musketeer", "ice_golem", "skeletons", "ice_spirit", "cannon", "fireball", "the_log"]
RG = ["royal_giant", "fisherman", "hunter", "electro_spirit", "fireball", "skeletons", "royal_ghost", "barbarian_barrel"]
LOGBAIT = ["goblin_barrel", "princess", "goblin_gang", "knight", "rocket", "ice_spirit", "inferno_tower", "the_log"]


@pytest.fixture
def toy():
    return BayesianDeckGuesser([(HOG26, 6), (RG, 3), (LOGBAIT, 1)])


def test_posterior_from_exact_deck_matches(toy):
    g = toy.guess(["hog_rider"])
    assert g.matching_weight == 6
    assert g.likely_decks[0] == (tuple(sorted(HOG26)), 1.0)
    assert g.card_probs["musketeer"] > 0.55          # strongly implied by the only matching deck
    assert "hog_rider" not in g.card_probs


def test_shared_cards_get_mixture_probability(toy):
    g = toy.guess(["skeletons", "fireball"])         # in both Hog and RG decks (6:3)
    assert g.likely_decks[0][1] == pytest.approx(6 / 9)
    assert g.card_probs["musketeer"] > g.card_probs["fisherman"] > g.card_probs.get("princess", 0)


def test_probabilities_fill_remaining_slots(toy):
    for revealed in ([], ["hog_rider"], ["the_log", "ice_spirit"], ["royal_giant", "hunter", "fireball"]):
        g = toy.guess(revealed)
        assert sum(g.card_probs.values()) == pytest.approx(8 - len(revealed))
        assert all(0 <= p <= 1 for p in g.card_probs.values())


def test_off_meta_deck_falls_back_to_cooccurrence(toy):
    g = toy.guess(["hog_rider", "princess"])          # no deck has both
    assert g.matching_weight == 0 and g.likely_decks == []
    assert sum(g.card_probs.values()) == pytest.approx(6)
    assert g.card_probs["musketeer"] > 0 and g.card_probs["goblin_barrel"] > 0


def test_unknown_card_does_not_crash(toy):
    g = toy.guess(["brand_new_card_2027"])
    assert sum(g.card_probs.values()) == pytest.approx(7)


def test_full_deck_revealed(toy):
    g = toy.guess(HOG26)
    assert g.card_probs == {} and g.likely_decks[0][1] == 1.0


def test_exclude(toy):
    assert "musketeer" not in toy.guess(["hog_rider"], exclude=["musketeer"]).card_probs


def test_in_hand_probs(toy):
    g = toy.guess(["hog_rider", "cannon", "the_log", "fireball"])   # 4 unknown cards left
    hp = g.in_hand_probs(unknown_hand_slots=2)
    assert sum(hp.values()) == pytest.approx(2.0)
    assert g.in_hand_probs(0) == {c: 0.0 for c in g.card_probs}


def test_real_meta_prior():
    g = default_guesser()
    if g is None:
        pytest.skip("configs/meta_decks_hf.json missing")
    hog = g.guess(["hog_rider", "ice_golem"])
    assert hog.top(1)[0][1] > 0.8                    # Hog 2.6 is the dominant hog deck in the data
    rg = dict(g.guess(["royal_giant"]).card_probs)
    assert rg["fisherman"] > rg.get("hog_rider", 0)


def test_known_costs():
    costs = known_costs()
    if not costs:
        pytest.skip("configs/card_costs_hf.json missing")
    assert costs["hog_rider"] == 4 and costs["the_log"] == 2 and len(costs) > 100
