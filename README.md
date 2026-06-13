I have a project repo with the following planning docs already in the root:
PRD.md, ARCHITECTURE.md, TASKS.md, TECH_STACK.md, EVAL_SPEC.md, README.md.

Read all of them first to understand the full scope.

Then execute TASKS.md in order, starting with Phase 1 (T1.1 through T1.9).
For each task:
1. Implement the code as specified in ARCHITECTURE.md and TECH_STACK.md.
2. Run it to verify it works (fix errors until it does).
3. Mark the task as done in TASKS.md.
4. Commit with a message following the convention in TECH_STACK.md.

Do not skip ahead to Phase 2 until all Phase 1 tasks pass their smoke tests
and `evaluation/results_phase1.md` is produced with real numbers (not placeholders).

After Phase 1 is complete, stop and summarize results before proceeding to Phase 2.