# AGENTS for ai_tools

## Superpowers Workflow (Mandatory)

For ALL work in this repository, you MUST follow the superpowers workflow end-to-end. Do not bypass skills or improvise your own process when a skill applies.

### 1. Superpowers Guardrail
- Always apply `using-superpowers` first.
- Before answering, coding, or even asking clarifying questions, check which skills apply to the current task.
- If there is any chance a skill is relevant, you MUST invoke and follow it.

### 2. Design & Requirements
- Use `brainstorming` for any creative work: new features, behavior changes, refactors, or architecture changes.
- Understand goals, constraints, and success criteria through one-question-at-a-time dialogue.
- Explore 2–3 approaches and pick a recommended one before implementation.

### 3. Workflow Orchestration
- Use `start-workflow` to decide the overall development path (feature vs bugfix vs refactor) and which skills to chain next.

### 4. Workspace Isolation
- Use `using-git-worktrees` to create an isolated worktree/branch before significant changes.
- Do all feature or bugfix work in this isolated workspace.

### 5. Planning
- Use `writing-plans` after design is agreed.
- Create a detailed implementation plan in `docs/plans/YYYY-MM-DD-<feature-name>.md`.
- Each step should be small (2–5 minutes), with explicit files, code, and commands.

### 6. Implementation (TDD)
- Use `test-driven-development` for any feature or bugfix implementation.
- Sequence: write failing test → run and confirm failure → write minimal code → run and confirm pass → refactor with tests green.

### 7. Executing the Plan
- Use `subagent-driven-development` when executing the plan in this session (task-by-task with review between tasks), OR
- Use `executing-plans` when executing the plan in a separate session.

### 8. Debugging
- Use `systematic-debugging` whenever a bug, test failure, or unexpected behavior appears.
- Reproduce, gather evidence, hypothesize, test hypotheses, fix, then re-verify.

### 9. Verification
- Use `verification-before-completion` before claiming any work is “done” or “fixed”.
- Run the plan’s verification commands and confirm outputs match expectations before stating success.

### 10. Code Review & Integration
- Use `requesting-code-review` when work is ready for review.
- Use `receiving-code-review` to process feedback rigorously and safely.
- Use `finishing-a-development-branch` to merge/integrate the work and clean up worktrees/branches.

## Rule

When working in this repo, you MUST follow this workflow strictly whenever the corresponding situation arises. Skills are not optional; they define the required process.
