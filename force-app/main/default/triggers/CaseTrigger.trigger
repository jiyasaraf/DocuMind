trigger CaseTrigger on Case (After update) {
	if(trigger.isAfter && trigger.isUpdate) {
        CaseTriggerHandler.handleCaseCreation(Trigger.new);
    }
}
