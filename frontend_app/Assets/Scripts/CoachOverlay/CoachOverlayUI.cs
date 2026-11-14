using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace CRProCoach.CoachOverlay
{
    /// <summary>
    /// Displays real-time win-rate trajectory curve, blunder warnings,
    /// and provides an instant "Takeover / Replay" button to retry critical situations.
    /// </summary>
    public class CoachOverlayUI : MonoBehaviour
    {
        [Header("UI References")]
        public Text winRateText;
        public Text oppElixirText;
        public Button takeoverReplayButton;
        public GameObject blunderAlertBanner;

        private float _currentWinRate = 0.50f;
        private float _estimatedOppElixir = 5.0f;

        void Start()
        {
            if (takeoverReplayButton != null)
            {
                takeoverReplayButton.onClick.AddListener(OnTakeoverClicked);
            }
            UpdateHUD();
        }

        public void SetCoachTelemetry(float winRate, float oppElixir, bool isBlunderWarning = false)
        {
            _currentWinRate = Mathf.Clamp01(winRate);
            _estimatedOppElixir = Mathf.Clamp(oppElixir, 0f, 10f);

            if (blunderAlertBanner != null)
            {
                blunderAlertBanner.SetActive(isBlunderWarning);
            }
            UpdateHUD();
        }

        private void UpdateHUD()
        {
            if (winRateText != null)
            {
                winRateText.text = $"Win Rate: {(_currentWinRate * 100f):F1}%";
                winRateText.color = _currentWinRate >= 0.5f ? Color.green : Color.red;
            }

            if (oppElixirText != null)
            {
                oppElixirText.text = $"Opp Elixir: ~{_estimatedOppElixir:F1} / 10";
            }
        }

        public void OnTakeoverClicked()
        {
            Debug.Log("[CoachOverlay] User requested instant takeover! Resetting board to blunder timestamp.");
            // Signal WebSocket / ServerBridge to pause live recording and enter interactive sandbox
        }
    }
}
