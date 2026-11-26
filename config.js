/* ============================================================
   HOST SETUP
   This uses the same Supabase project as the phd_games repository.
   Before the first class, run supabase.sql once (SQL Editor) so the
   results table accepts the three new game codes: rnn, gate, mdp.

   SESSION: a label for this class. Change it before each lecture
   (for example "qip-L20", "qip-L21", "qip-L24") so the room screen
   starts empty. The room screen can also be opened with
   results.html?session=qip-L21 without editing this file.
   ============================================================ */
window.GAME_CONFIG = {
  SUPABASE_URL: "https://bphschorkhgwnjcxzjkc.supabase.co",
  SUPABASE_KEY: "sb_publishable_P3YBd9zZ57yasBvEYLLy5g_JiLW_8mc",
  SESSION: "qip-L20"
};
