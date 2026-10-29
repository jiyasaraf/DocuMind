trigger AgentWorkTrigger on AgentWork(
    before insert,
    after insert,
    after update
) {
    wkdw.AgentWorkTriggerHandlerGlobalProxy.handleTrigger(
        Trigger.newMap,
        Trigger.oldMap,
        Trigger.operationType
    );
}
