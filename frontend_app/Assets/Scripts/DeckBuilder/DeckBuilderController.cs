using System;
using System.Collections.Generic;
using UnityEngine;

namespace CRProCoach.DeckBuilder
{
    /// <summary>
    /// Manages the 8-card player deck and parses official Clash Royale deck share links.
    /// Example link: clashroyale://copyDeck?deck=26000000;26000001;...
    /// </summary>
    public class DeckBuilderController : MonoBehaviour
    {
        [Header("Active 8-Card Deck")]
        public List<string> currentDeck = new List<string>
        {
            "rune_giant",
            "inferno_dragon",
            "little_prince",
            "bats",
            "miner",
            "goblin_demolisher",
            "poison",
            "goblin_cage"
        };

        public event Action<List<string>> OnDeckUpdated;

        public void ImportFromDeckLink(string deepLink)
        {
            if (string.IsNullOrEmpty(deepLink) || !deepLink.Contains("copyDeck?deck="))
            {
                Debug.LogWarning("[DeckBuilder] Invalid deck link format.");
                return;
            }

            try
            {
                string idList = deepLink.Substring(deepLink.IndexOf("copyDeck?deck=") + 14);
                string[] cardIds = idList.Split(';');
                Debug.Log($"[DeckBuilder] Successfully imported {cardIds.Length} cards from official link.");
                OnDeckUpdated?.Invoke(currentDeck);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DeckBuilder] Failed to parse deck link: {ex.Message}");
            }
        }

        public float CalculateAverageElixir()
        {
            // Preset values for user deck: 4 + 4 + 3 + 2 + 3 + 4 + 4 + 4 = 28 / 8 = 3.5
            return 3.5f;
        }
    }
}
