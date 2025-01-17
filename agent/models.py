'''from django.db import models

class AgentRole(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'agent_roles'

class AgentTask(models.Model):
    role = models.ForeignKey(AgentRole, on_delete=models.CASCADE)
    task_input = models.TextField()
    task_output = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=20, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'agent_tasks'
'''