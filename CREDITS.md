# Credits

## Battle board art

The pre-rendered sprites in `frontend_app/qt/assets/sprites/` (characters, towers, castles, trees, rocks) are
rendered from these 3D model packs by **Kay Lousberg** (www.kaylousberg.com), released under
**Creative Commons Zero (CC0 1.0)**: free for personal, educational and commercial use, credit not required
(given here with thanks):

* KayKit Character Pack: Adventurers 1.0: https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0
* KayKit Character Pack: Skeletons 1.0: https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Skeletons-1.0
* KayKit Medieval Hexagon Pack 1.0: https://github.com/KayKit-Game-Assets/KayKit-Medieval-Hexagon-Pack-1.0

The ground (grass, stone plazas, dirt lanes, water) and the plank bridges are generated procedurally in
`tools/art/blender_render.py`. Rebuild everything with `tools/art/build_sprites.py` (needs Blender); the build
refuses any pack whose `LICENSE.txt` is not CC0.

CRbot uses no images, characters, icons, fonts or sounds from Supercell or Clash Royale.
