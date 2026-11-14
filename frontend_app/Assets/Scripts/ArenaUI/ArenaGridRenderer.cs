using System;
using System.Collections.Generic;
using UnityEngine;

namespace CRProCoach.ArenaUI
{
    /// <summary>
    /// Renders the 18x32 Clash Royale arena grid with 36x64 half-tile resolution.
    /// Manages drag-and-drop card placement and Level 11 / 16 multiplier switching.
    /// </summary>
    public class ArenaGridRenderer : MonoBehaviour
    {
        public const int GridX = 36;
        public const int GridY = 64;
        public const float TileWidth = 0.5f;

        [Header("Display Settings")]
        public bool showGridGizmos = true;
        public int currentLevelMode = 16; // 11 or 16

        [Header("Ghost Recommendation")]
        public GameObject ghostIndicatorPrefab;
        private GameObject _activeGhostTile;

        void Start()
        {
            Debug.Log($"[ArenaGridRenderer] Initialized with Level {currentLevelMode} stats.");
        }

        public void SetLevelMode(int level)
        {
            if (level == 11 || level == 16)
            {
                currentLevelMode = level;
                Debug.Log($"[ArenaGridRenderer] Switched combat scale to Level {currentLevelMode}.");
            }
        }

        public Vector2 CellIdToWorldPos(int cellId)
        {
            int gx = cellId % GridX;
            int gy = cellId / GridX;
            float worldX = (gx - GridX / 2.0f + 0.5f) * TileWidth;
            float worldY = (gy - GridY / 2.0f + 0.5f) * TileWidth;
            return new Vector2(worldX, worldY);
        }

        public int WorldPosToCellId(Vector2 worldPos)
        {
            int gx = Mathf.Clamp(Mathf.FloorToInt(worldPos.x / TileWidth + GridX / 2.0f), 0, GridX - 1);
            int gy = Mathf.Clamp(Mathf.FloorToInt(worldPos.y / TileWidth + GridY / 2.0f), 0, GridY - 1);
            return gy * GridX + gx;
        }

        public void ShowGhostRecommendation(int cellId, string cardName)
        {
            Vector2 pos = CellIdToWorldPos(cellId);
            if (_activeGhostTile == null && ghostIndicatorPrefab != null)
            {
                _activeGhostTile = Instantiate(ghostIndicatorPrefab, pos, Quaternion.identity);
            }
            else if (_activeGhostTile != null)
            {
                _activeGhostTile.transform.position = pos;
                _activeGhostTile.SetActive(true);
            }
            Debug.Log($"[Coach Ghost] Recommending {cardName} at Cell {cellId} ({pos.x}, {pos.y})");
        }

        public void ClearGhost()
        {
            if (_activeGhostTile != null)
            {
                _activeGhostTile.SetActive(false);
            }
        }
    }
}
