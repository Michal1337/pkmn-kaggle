#pragma once
#include "All.h"

// Native-encode binary obs (M2): redacted per-player obs -> flat int32 buffer, mirroring
// ToJsonApi/Current/PlayerJson/PokemonJson with playerIndex = state.selectPlayer. Order MUST match
// the Python reader. Returns #ints written (<= cap). Sections: A scalars, B counts+status,
// C card lists (hand/discard/stadium), D units (active+bench, both sides).
inline int WriteBinaryObs(const State& state, int* buf, int cap) {
  int n = 0;
  auto put = [&](long long v) { if (n < cap) buf[n] = (int)v; ++n; };
  const int me = state.selectPlayer;

  // --- A: global scalars (Current) ---
  put(state.turn); put(state.turnActionCount); put(me); put(state.firstPlayer);
  put(state.supporterPlayed); put(state.stadiumPlayed); put(state.energyPlayed);
  put(state.retreated); put(state.apiResult());
  // --- B: per-player counts + status (players 0 then 1) ---
  for (int p = 0; p < 2; ++p) {
    const PlayerState& ps = state.players[p];
    put(state.benchCapacity(p)); put(ps.deck.size()); put(ps.hand.size());
    put(ps.prize.size()); put(ps.trash.size()); put(ps.active.size()); put(ps.bench.size());
    put(ps.isPoisoned()); put(ps.burned);
    put(ps.badStatus == BadStatusType::Asleep); put(ps.badStatus == BadStatusType::Paralyzed);
    put(ps.badStatus == BadStatusType::Confused);
  }

  auto put_card = [&](CardRef ref) {
    const Card& card = state.getCard(ref);
    put(card.getMaster().cardId); put(ref.cardIndex); put(card.playerIndex);
  };
  auto put_list = [&](const auto& list) {
    put((int)list.size());
    for (int i = 0; i < (int)list.size(); ++i) put_card(list[i]);
  };

  // --- C: card lists (id, serial, playerIndex), length-prefixed ---
  put_list(state.players[me].hand);   // self hand
  put_list(state.players[0].trash);   // discard 0
  put_list(state.players[1].trash);   // discard 1
  put_list(state.stadium);            // stadium

  // --- D: units (active+bench, both sides). Each slot: 0=null(facedown) | 1 + pokemon fields ---
  auto put_pokemon = [&](CardRef ref) {
    const Card& card = state.getCard(ref);
    if (card.reverse) { put(0); return; }          // facedown (addName=false in the per-player view)
    put(1);
    put(card.getMaster().cardId); put(ref.cardIndex); put(card.playerIndex);
    put(state.getHp(card)); put(state.getMaxHp(card)); put(card.appear);
    auto& energies = state.game->energyList;
    state.getEnergies(card.playerIndex, ref, energies);
    put((int)energies.size());
    for (int i = 0; i < (int)energies.size(); ++i) put(EnergyTypeIndex(energies[i]));
    auto& ecards = state.game->cardList; state.getEnergyCards(ref, ecards); put_list(ecards);
    auto tools = state.getAttachedToolRef(card); put_list(tools);
    auto pre = state.getPreEvolutions(card); put_list(pre);
  };
  auto put_pokelist = [&](const auto& list) {
    put((int)list.size());
    for (int i = 0; i < (int)list.size(); ++i) put_pokemon(list[i]);
  };
  for (int p = 0; p < 2; ++p) {
    put_pokelist(state.players[p].active);
    put_pokelist(state.players[p].bench);
  }

  // --- E: select + options (SelectJson). option = type + param0..4 (Python replicates the switch) ---
  put(std::max(0, (int)state.selectType - 1));      // type (enum-1 wire offset)
  put(std::max(0, (int)state.selectContext - 1));   // context
  put(state.selectMin);                             // minCount
  put(state.selectMax);                             // maxCount
  put(state.remainDamageCounter);
  put(state.remainEnergyCost);
  put((int)state.options.size());
  for (int i = 0; i < (int)state.options.size(); ++i) {
    const SelectOption& o = state.options[i];
    put((int)o.type);
    put(o.param0); put(o.param1); put(o.param2); put(o.param3); put(o.param4);
  }
  // deck (deck-search shown subset): count>=0 + cards, or -1 == null
  if (state.selectDeck) put_list(state.players[me].deck);
  else put(-1);
  // contextCard / effect: 1 + card, or 0 == null
  if (state.contextCard.isNull()) put(0); else { put(1); put_card(state.contextCard); }
  if (state.onEffect()) { put(1); put_card(state.getEffectCard().card); } else put(0);

  // --- F: looking (for zone_card area 12 option-src resolution). Mirror Current's redaction. ---
  // -1 count = None; else count then count*(id,serial,playerIndex) with id==-1 for a facedown null.
  int lp = state.lookingPlayer;
  int lsz = (int)state.looking.size();
  if (lsz == 0 || (lp != me && lp != 2 && me != 2)) {
    if (lsz != 0 && lp >= 3 && lp == me + 3) {
      put(lsz);
      for (int k = 0; k < lsz; ++k) { put(-1); put(-1); put(-1); }
    } else {
      put(-1);
    }
  } else {
    put_list(state.looking);
  }

  // --- G: logs (delta since logIndex[me]) for the GameTracker -- PEEK (non-consuming; the caller
  // advances via AdvanceLog once per decision). Per log 7 ints: type, playerIndex, cardId, serial,
  // fromArea, toArea, attackId (-1 = absent/redacted). Mirror LogJson's MoveCard(6)->Reverse(7)
  // redaction (the ONLY one the tracker sees; Play/Attach/Evolve/Devolve/MoveAttached/Attack are
  // public). Skip logType > Result (as LogsJson does).
  int lstart = state.logIndex[me];
  int ltotal = (int)state.logs.size();
  int lcnt = 0;
  for (int i = lstart; i < ltotal; ++i)
    if (state.logs[i].logType <= LogType::Result) ++lcnt;
  put(lcnt);
  for (int i = lstart; i < ltotal; ++i) {
    const Log& lg = state.logs[i];
    if (lg.logType > LogType::Result) continue;
    int lt = (int)lg.logType;
    int etype = lt, cid = -1, ser = -1, fA = -1, tA = -1, aid = -1;
    int pid = lg.param.size() > 0 ? lg.param[0] : -1;
    if (lt == 6) {                                       // MoveCard: redact by openType param[5]
      bool vis = (lg.param[5] == 0) || (lg.param[5] == 1 && lg.param[0] == me)
              || (lg.param[5] == 3 && me == 0) || (lg.param[5] == 4 && me == 1) || (me == 2);
      if (vis) { cid = lg.param[1]; ser = lg.param[2]; fA = lg.param[3]; tA = lg.param[4]; }
      else { etype = 7; fA = lg.param[3]; tA = lg.param[4]; }        // -> MoveCardReverse (no id/serial)
    } else if (lt == 4) {                                // Draw: id visible to the DRAWING player
      // only (ApiJson emits DrawReverse(5) to the other side). The v2.3 deck-top tracker needs
      // this id to verify its stack against the card actually drawn.
      if (lg.param[0] == me || me == 2) { cid = lg.param[1]; ser = lg.param[2]; }
      else { etype = 5; }
    } else if (lt == 7) { fA = lg.param[1]; tA = lg.param[2]; }      // native MoveCardReverse
    else if (lt == 10) { cid = lg.param[1]; ser = lg.param[2]; }    // Play
    else if (lt == 11 || lt == 12 || lt == 13 || lt == 14) { cid = lg.param[1]; ser = lg.param[2]; }  // reveal
    else if (lt == 15) { cid = lg.param[1]; ser = lg.param[2]; aid = lg.param[3]; }                   // Attack
    put(etype); put(pid); put(cid); put(ser); put(fA); put(tA); put(aid);
  }
  return n;
}

extern "C" {
#ifdef _MSC_VER
  __declspec(dllexport)
#else
  __attribute__((visibility("default")))
#endif
  int GetBinaryObs(ApiData* data, int* buf, int cap) {
    if (data->apiDataType != 1) return -1;
    return WriteBinaryObs(data->state, buf, cap);
  }

#ifdef _MSC_VER
  __declspec(dllexport)
#else
  __attribute__((visibility("default")))
#endif
  // Consume the log delta for the current player: logIndex[me] = logs.size(). The native env calls
  // this ONCE per decision (after GameTracker.update_native reads the peeked delta) so the next
  // decision's GetBinaryObs peek returns only the new logs -- mirrors GetBattleData's logStart.
  int AdvanceLog(ApiData* data) {
    if (data->apiDataType != 1) return -1;
    data->state.logIndex[data->state.selectPlayer] = (int)data->state.logs.size();
    return 0;
  }
}
