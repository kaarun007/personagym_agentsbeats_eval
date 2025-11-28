# personagym_agentsbeats_eval
Current State:

Here is the explanation of the code sequence in PersonaGym, starting from run.py:

High-Level Overview
The code evaluates "personas" (simulated characters) by placing them in specific settings, asking them challenging questions, and then grading their responses using LLM-based judges against defined rubrics.

PersonaGym Execution Flow
Evaluation Process
Start: run.py
main()
select_settingsChoose Scenarios
gen_questionsGenerate Challenges
gen_answersPersona Responds
score_answersEvaluate Responses
format_rubricsPrepare Prompts
gen_score_examplesCreate Few-Shot Examples
score_rubricsLLM Grading
Calculate Average Scores
Save Results & Scores
End
