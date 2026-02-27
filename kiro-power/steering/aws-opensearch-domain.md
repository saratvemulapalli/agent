---
title: "AWS OpenSearch Domain Deployment for Agentic Search"
inclusion: manual
---

# AWS OpenSearch Domain Deployment Guide (Agentic Search)

This guide covers deploying your local OpenSearch search strategy to AWS OpenSearch Domain (managed cluster).

## When to Use OpenSearch Domain

Use OpenSearch Domain for:
- **Agentic Search applications** (required)
- Workloads requiring advanced plugins
- Applications needing fine-grained control over cluster configuration
- High-performance requirements with dedicated resources
- Custom plugin installations
- Advanced security configurations

## Why Domain for Agentic Search?

Agentic search requires:
- Advanced query capabilities and custom scoring
- Complex aggregations and analytics
- Plugin support for specialized functionality
- Predictable performance with dedicated resources
- Fine-grained cluster tuning

OpenSearch Serverless does not support these requirements.

## Prerequisites

Before starting Phase 5 deployment:
1. AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
2. Appropriate IAM permissions for OpenSearch Service
3. Successful Phase 4 execution with local OpenSearch running
4. Search strategy manifest file created in Phase 3
5. VPC and subnet configuration (for VPC deployment)

## Deployment Steps

### Step 1: Create OpenSearch Domain

Use the AWS MCP server to create a domain:

```
aws opensearch create-domain
  --domain-name <domain-name>
  --engine-version OpenSearch_2.11 (or latest)
  --cluster-config <cluster-config>
  --ebs-options <ebs-options>
  --access-policies <access-policy>
  --node-to-node-encryption-options Enabled=true
  --encryption-at-rest-options Enabled=true
  --domain-endpoint-options EnforceHTTPS=true
```

### Step 2: Configure Cluster Topology

Choose instance types and cluster size based on workload:

**Development/Testing:**
- Instance type: t3.small.search or t3.medium.search
- Data nodes: 1-2
- Master nodes: Not required for small clusters

**Production:**
- Instance type: r6g.large.search or larger (memory-optimized for agentic workloads)
- Data nodes: 3+ (for high availability)
- Dedicated master nodes: 3 (recommended for production)
- UltraWarm nodes: Optional for cost optimization of older data

### Step 3: Configure Storage

Set up EBS volumes:

```json
{
  "EBSEnabled": true,
  "VolumeType": "gp3",
  "VolumeSize": 100,
  "Iops": 3000,
  "Throughput": 125
}
```

Size based on:
- Expected document count
- Index size from local testing
- Growth projections
- Replica requirements

### Step 4: Configure Network Access

Choose access configuration:

**Public Access (Development):**
- Set access policies with IP restrictions
- Use fine-grained access control

**VPC Access (Production - Recommended):**
- Deploy domain within VPC
- Configure security groups
- Set up VPC endpoints if needed
- Ensure proper subnet configuration

### Step 5: Enable Fine-Grained Access Control

Configure authentication and authorization:

```
aws opensearch update-domain-config
  --domain-name <domain-name>
  --advanced-security-options Enabled=true,InternalUserDatabaseEnabled=true,MasterUserOptions={...}
```

Set up:
- Master user credentials
- Role-based access control
- Backend roles mapping
- Index-level permissions

### Step 6: Wait for Domain to be Active

Poll domain status until active:

```
aws opensearch describe-domain
  --domain-name <domain-name>
```

Wait for:
- Processing: false
- DomainStatus: Active
- Endpoint available

This typically takes 10-15 minutes.

### Step 7: Migrate Index Configuration

Using the opensearch-mcp-server tools:

1. Get the local index configuration from the manifest
2. Create the index on the domain endpoint
3. Include all mappings, settings, and configurations
4. Configure replicas for high availability (typically 1-2 replicas)

### Step 8: Deploy ML Models

For agentic search with embeddings:

1. Deploy models to the OpenSearch cluster:
   - Use pretrained models from OpenSearch model repository
   - Or deploy custom models
2. Configure model settings (memory, inference threads)
3. Test model inference performance
4. Update pipeline configurations to use deployed models

### Step 9: Create Ingest Pipelines

Recreate ingest pipelines on the domain:

1. Get pipeline definitions from local setup
2. Create pipelines using opensearch-mcp-server
3. Attach pipelines to the index
4. Configure processors for agentic search requirements

### Step 10: Configure Search Pipelines (if applicable)

For advanced agentic search features:

1. Create search pipelines for query processing
2. Configure query rewriting and expansion
3. Set up custom scoring and ranking
4. Enable search relevance tuning

### Step 11: Index Sample Documents

Index test documents to verify the setup:

1. Use the same sample documents from Phase 1
2. Verify embeddings and processing
3. Test agentic search queries
4. Validate performance and relevance

### Step 12: Configure Monitoring and Alerting

Set up observability:

1. Enable CloudWatch logs:
   - Index slow logs
   - Search slow logs
   - Error logs
   - Audit logs
2. Create CloudWatch alarms for:
   - Cluster health
   - CPU and memory utilization
   - Storage space
   - JVM pressure
3. Set up SNS notifications

### Step 13: Provide Access Information

Give the user:
- Domain endpoint URL
- Domain ARN
- OpenSearch Dashboards URL
- Master user credentials (securely)
- Sample agentic search queries to test
- Cost estimation based on instance types and configuration

## Cost Considerations

OpenSearch Domain pricing:
- Instance hours (varies by instance type)
- EBS storage (GB-month)
- Data transfer
- Snapshot storage (if enabled)

**Cost optimization tips:**
- Use reserved instances for production (up to 30% savings)
- Right-size instances based on actual usage
- Use UltraWarm for infrequently accessed data
- Enable automated snapshots to S3

Typical monthly cost for small production cluster:
- 3x r6g.large.search: ~$400-500/month
- 300GB EBS storage: ~$30/month
- Total: ~$450-550/month

## Security Best Practices

1. **Network Security:**
   - Deploy in VPC for production
   - Use security groups to restrict access
   - Enable VPC Flow Logs

2. **Access Control:**
   - Enable fine-grained access control
   - Use IAM roles for application access
   - Implement least-privilege policies
   - Rotate credentials regularly

3. **Encryption:**
   - Enable encryption at rest
   - Enable node-to-node encryption
   - Enforce HTTPS for all connections

4. **Monitoring:**
   - Enable all CloudWatch logs
   - Set up alerting for security events
   - Regular security audits

## High Availability and Disaster Recovery

1. **Multi-AZ Deployment:**
   - Enable zone awareness
   - Distribute nodes across 3 AZs
   - Configure standby replicas

2. **Backup Strategy:**
   - Enable automated snapshots
   - Configure snapshot repository in S3
   - Test restore procedures
   - Retain snapshots based on compliance requirements

3. **Disaster Recovery:**
   - Document recovery procedures
   - Set up cross-region replication if needed
   - Define RTO and RPO targets

## Performance Tuning for Agentic Search

1. **Index Settings:**
   - Optimize refresh interval
   - Configure appropriate shard count
   - Tune merge policies

2. **Query Optimization:**
   - Use query caching
   - Optimize aggregations
   - Implement request caching

3. **Resource Allocation:**
   - Monitor JVM heap usage
   - Adjust circuit breakers if needed
   - Configure thread pools for workload

## Troubleshooting

Common issues:

- **Domain creation fails**: Check service quotas, VPC configuration, IAM permissions
- **Cluster health yellow/red**: Check shard allocation, storage space, node health
- **Slow queries**: Review slow logs, optimize queries, check resource utilization
- **Model deployment fails**: Verify ML plugin enabled, check memory allocation
- **Access denied**: Verify fine-grained access control settings, IAM policies

## Next Steps

After successful deployment:

1. Update application code to use the domain endpoint
2. Implement connection pooling and retry logic
3. Set up comprehensive monitoring dashboards
4. Configure automated backups
5. Plan for capacity scaling
6. Document operational procedures
7. Train team on OpenSearch Dashboards
8. Implement performance testing and optimization
