stateDiagram-v2
    [*] --> Perceive
    
    Perceive --> AnalyzeQuery
    AnalyzeQuery --> CheckAmbiguity
    
    CheckAmbiguity --> NeedsClarification: Ambiguous OR<br/>Bounds Too Broad
    CheckAmbiguity --> Plan: Clear & Feasible
    
    NeedsClarification --> PAUSED_FOR_CLARIFICATION
    
    state PAUSED_FOR_CLARIFICATION {
        [*] --> SaveCheckpoint
        SaveCheckpoint --> WaitForUser
        WaitForUser --> UserResponds: User Input
        WaitForUser --> Timeout: No Response
        UserResponds --> RestoreCheckpoint
        RestoreCheckpoint --> SummarizeProgress
        SummarizeProgress --> [*]
    }
    
    PAUSED_FOR_CLARIFICATION --> Perceive: Resume with New Info
    PAUSED_FOR_CLARIFICATION --> [*]: Timeout/Abandon
    
    Plan --> DecomposeGoals: Complex Task
    Plan --> SimpleExecution: Simple Task
    
    DecomposeGoals --> SubtaskPlanning
    SubtaskPlanning --> CheckPlanFeasibility
    
    CheckPlanFeasibility --> NeedsClarification: Missing Critical Info
    CheckPlanFeasibility --> DependencyAnalysis: Plan Complete
    
    DependencyAnalysis --> Act
    SimpleExecution --> Act
    
    Act --> ToolSelection
    ToolSelection --> ToolExecution
    
    state ToolExecution {
        [*] --> ExecuteTool
        ExecuteTool --> WaitForCompletion
        WaitForCompletion --> ResultValidation
    }
    
    ToolExecution --> ResultValidation
    
    ResultValidation --> QueryRefinement: Poor Results & Retries Left
    ResultValidation --> Evaluate: Good Results
    ResultValidation --> Evaluate: Max Retries Reached
    
    QueryRefinement --> ToolExecution
    
    Evaluate --> ConfidenceCheck
    ConfidenceCheck --> Reflect: Sufficient Evidence
    ConfidenceCheck --> Plan: Need More Data
    
    Reflect --> MemoryDecision
    MemoryDecision --> RecordMemory: Valuable Learning
    MemoryDecision --> SkipMemory: No New Insight
    
    RecordMemory --> NextStep
    SkipMemory --> NextStep
    
    NextStep --> Plan: More Subtasks
    NextStep --> CompileReport: All Complete
    
    CompileReport --> [*]
    
    note right of PAUSED_FOR_CLARIFICATION
        In-flight operations complete
        Results held in checkpoint
        Full state persisted to PostgreSQL
    end note
    
    note right of ToolExecution
        Tools continue to completion
        even if clarification needed
    end note
