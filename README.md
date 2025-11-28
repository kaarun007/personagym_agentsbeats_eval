# personagym_agentsbeats_eval
Current State:

Here is the explanation of the code sequence in PersonaGym, starting from run.py:

High-Level Overview
The code evaluates "personas" (simulated characters) by placing them in specific settings, asking them challenging questions, and then grading their responses using LLM-based judges against defined rubrics.

graph TD
    Start([Start: run.py]) --> Main{main()}
    Main --> SelectSettings[select_settings<br/>Choose Scenarios]
    SelectSettings --> GenQuestions[gen_questions<br/>Generate Challenges]
    GenQuestions --> GenAnswers[gen_answers<br/>Persona Responds]
    GenAnswers --> ScoreAnswers[score_answers<br/>Evaluate Responses]
    
    subgraph Evaluation_Loop [Evaluation Process]
        ScoreAnswers --> FormatRubrics[format_rubrics<br/>Prepare Prompts]
        FormatRubrics --> GenScoreExamples[gen_score_examples<br/>Create Few-Shot Examples]
        GenScoreExamples --> ScoreRubrics[score_rubrics<br/>LLM Grading]
        ScoreRubrics --> Aggregate[Calculate Average Scores]
    end
    
    Aggregate --> SaveResults[Save Results & Scores]
    SaveResults --> End([End])
    
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Evaluation_Loop fill:#e1f5fe,stroke:#01579b,stroke-width:2px
