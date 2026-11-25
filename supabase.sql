-- ============================================================
--  Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
--
--  If you already ran phd_games/supabase.sql in this project, only the
--  ALTER statements in part 2 are needed. Running the whole file again is
--  harmless: every statement is written to be repeatable.
-- ============================================================

-- 1. The results table (same as in phd_games)
create table if not exists public.plays (
  id          uuid primary key default gen_random_uuid(),
  created_at  timestamptz not null default now(),
  session     text not null check (char_length(session) between 1 and 40),
  game        text not null,
  nickname    text not null check (char_length(nickname) between 1 and 30),
  score       integer not null check (score between 0 and 100),
  total       integer not null check (total between 1 and 100),
  answers     jsonb,
  prompt      text check (prompt is null or char_length(prompt) <= 4000),
  weak_prompt text check (weak_prompt is null or char_length(weak_prompt) <= 120)
);
create index if not exists plays_session_game_idx on public.plays (session, game, created_at desc);

-- 2. Allow the three QIPPG103 games alongside the PhD games
alter table public.plays drop constraint if exists plays_game_check;
alter table public.plays add constraint plays_game_check
  check (game in ('ref', 'hom', 'duel', 'rnn', 'gate', 'mdp'));

-- 3. Players may add a result and read results; nothing else
alter table public.plays enable row level security;
drop policy if exists "players can submit results" on public.plays;
create policy "players can submit results" on public.plays for insert to anon with check (true);
drop policy if exists "anyone can read results" on public.plays;
create policy "anyone can read results" on public.plays for select to anon using (true);
revoke all on public.plays from anon;
grant select, insert on public.plays to anon;

-- To clear a class afterwards:
--   delete from public.plays where session = 'qip-L20';
