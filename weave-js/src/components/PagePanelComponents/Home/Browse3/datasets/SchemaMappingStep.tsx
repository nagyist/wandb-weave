import {Box, Paper, Stack, Typography} from '@mui/material';
import React, {useCallback, useEffect, useMemo} from 'react';

import {Select} from '../../../../Form/Select';
import {Icon} from '../../../../Icon';
import {LoadingDots} from '../../../../LoadingDots';
import {useWFHooks} from '../pages/wfReactInterface/context';
import {
  ObjectVersionSchema,
  WeaveObjectVersionKey,
} from '../pages/wfReactInterface/wfDataModelHooksInterface';
import {DataPreviewTooltip} from './DataPreviewTooltip';
import {ACTION_TYPES, useDatasetDrawer} from './DatasetDrawerContext';
import {
  CallData,
  createTargetSchemaFromDenested,
  denestData,
  extractSourceSchema,
  FieldMapping,
  generateFieldPreviews,
} from './schemaUtils';

export interface SchemaMappingStepProps {
  selectedDataset: ObjectVersionSchema;
  selectedCalls: CallData[];
  entity: string;
  project: string;
  onMappingChange: (mappings: FieldMapping[]) => void;
  onDatasetObjectLoaded: (obj: any, rowsData?: any[]) => void;
  fieldMappings?: FieldMapping[];
  datasetObject?: any;
}

const typographyStyle = {fontFamily: 'Source Sans Pro'};

export const SchemaMappingStep: React.FC<SchemaMappingStepProps> = ({
  selectedDataset,
  selectedCalls,
  entity,
  project,
  onMappingChange,
  onDatasetObjectLoaded,
  fieldMappings = [],
}) => {
  const {useObjectVersion, useTableRowsQuery} = useWFHooks();
  const {dispatch} = useDatasetDrawer();

  const sourceSchema = useMemo(() => {
    return selectedCalls.length > 0 ? extractSourceSchema(selectedCalls) : [];
  }, [selectedCalls]);

  const objVersionKey = selectedDataset
    ? {
        scheme: 'weave' as WeaveObjectVersionKey['scheme'],
        weaveKind: 'object' as WeaveObjectVersionKey['weaveKind'],
        entity: selectedDataset.entity,
        project: selectedDataset.project,
        objectId: selectedDataset.objectId,
        versionHash: selectedDataset.versionHash,
        path: selectedDataset.path,
      }
    : null;
  const selectedDatasetObjectVersion = useObjectVersion({key: objVersionKey});

  useEffect(() => {
    if (selectedDataset && selectedDatasetObjectVersion.result?.val) {
      onDatasetObjectLoaded(selectedDatasetObjectVersion.result.val);
    }
  }, [
    selectedDataset,
    selectedDatasetObjectVersion.result,
    onDatasetObjectLoaded,
  ]);

  const tableDigest = selectedDatasetObjectVersion.result?.val?.rows
    ?.split('/')
    ?.pop();

  const tableRowsQuery = useTableRowsQuery({
    entity: entity || '',
    project: project || '',
    digest: tableDigest || '',
    limit: 10, // This is an arbitrary limit to prevent loading too much data.
  });

  // Memoize denested row data to avoid redundant processing
  const denestedRows = useMemo(() => {
    if (!tableRowsQuery.result?.rows) {
      return [];
    }

    return tableRowsQuery.result.rows.map(row => {
      return denestData(row.val);
    });
  }, [tableRowsQuery.result?.rows]);

  const targetSchema = useMemo(() => {
    if (!denestedRows.length) {
      return [];
    }
    return createTargetSchemaFromDenested(denestedRows);
  }, [denestedRows]);

  // Update target schema in context
  useEffect(() => {
    if (targetSchema.length > 0) {
      dispatch({type: ACTION_TYPES.SET_TARGET_SCHEMA, payload: targetSchema});
    }
  }, [targetSchema, dispatch]);

  // Pass rows data to parent when available
  useEffect(() => {
    // Only load rows data if we have a valid selected dataset
    // This prevents loading data from a previously selected dataset when in create new mode
    if (
      selectedDataset &&
      tableRowsQuery.result &&
      tableRowsQuery.result.rows
    ) {
      if (selectedDatasetObjectVersion.result?.val) {
        onDatasetObjectLoaded(
          selectedDatasetObjectVersion.result.val,
          tableRowsQuery.result.rows
        );
      }
    }
  }, [
    tableRowsQuery.result,
    selectedDatasetObjectVersion.result,
    selectedDataset,
    onDatasetObjectLoaded,
  ]);

  const handleMappingChange = useCallback(
    (targetField: string, sourceField: string | null) => {
      const existingMappings = fieldMappings || [];
      const newMappings = existingMappings.filter(
        m => m.targetField !== targetField
      );

      if (sourceField !== null && sourceField !== '') {
        newMappings.push({sourceField, targetField});
      }
      onMappingChange(newMappings);
    },
    [fieldMappings, onMappingChange]
  );

  const fieldPreviews = useMemo(() => {
    return generateFieldPreviews(sourceSchema, selectedCalls);
  }, [sourceSchema, selectedCalls]);

  const formatOptionLabel = useCallback(
    (option: {label: string; value: string}) => {
      if (option.value === '') {
        return option.label;
      }
      const previewData = fieldPreviews.get(option.value);
      return (
        <DataPreviewTooltip
          rows={previewData}
          tooltipProps={{
            placement: 'left-start',
            sx: {
              maxWidth: '800px',
            },
          }}>
          <div
            style={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              maxWidth: '100%',
            }}>
            {option.label}
          </div>
        </DataPreviewTooltip>
      );
    },
    [fieldPreviews]
  );

  const renderTargetField = useCallback(
    (fieldName: string) => {
      // Use the denested data array for previews
      const previewData = denestedRows.slice(0, 5).map(denested => {
        // Get only the top-level field we're interested in
        const topLevelKey = fieldName.split('.')[0];
        return {
          [fieldName]: denested[topLevelKey],
        };
      });

      return (
        <DataPreviewTooltip
          rows={previewData}
          tooltipProps={{
            placement: 'bottom-start',
            sx: {
              maxWidth: '800px',
            },
            componentsProps: {
              popper: {
                modifiers: [
                  {
                    name: 'offset',
                    options: {
                      offset: [0, 8],
                    },
                  },
                ],
              },
            },
          }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              padding: '4px 8px',
              borderRadius: 1,
              transition: 'all 0.2s ease',
              width: '100%',
              '&:hover': {
                backgroundColor: 'rgba(0, 0, 0, 0.04)',
              },
            }}>
            <Typography
              sx={{
                fontFamily: 'Monospace',
                fontWeight: 500,
                color: 'text.primary',
                fontSize: '14px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                width: '100%',
              }}>
              {fieldName}
            </Typography>
          </Box>
        </DataPreviewTooltip>
      );
    },
    [denestedRows]
  );

  if (
    tableRowsQuery.loading ||
    selectedDatasetObjectVersion.loading ||
    !targetSchema.length
  ) {
    return (
      <Stack spacing={1} sx={{mt: 2}}>
        <Typography component="div" sx={{...typographyStyle, fontWeight: 600}}>
          Field Mapping
        </Typography>
        <Paper
          variant="outlined"
          sx={{
            p: 2,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '56px',
          }}>
          <LoadingDots />
        </Paper>
      </Stack>
    );
  }

  if (!selectedCalls.length || !tableRowsQuery.result?.rows?.length) {
    return (
      <Stack spacing={1} sx={{mt: 2}}>
        <Typography component="div" sx={typographyStyle}>
          No data available to extract schema from.
        </Typography>
      </Stack>
    );
  }

  if (!sourceSchema.length || !targetSchema.length) {
    return (
      <Stack spacing={1} sx={{mt: 2}}>
        <Typography component="div" sx={typographyStyle}>
          No schema could be extracted from the data.
        </Typography>
      </Stack>
    );
  }

  return (
    <Stack spacing={'8px'} sx={{mt: '24px'}}>
      <Typography component="div" sx={{...typographyStyle, fontWeight: 600}}>
        Configure field mapping
      </Typography>
      <Typography
        component="div"
        sx={{
          ...typographyStyle,
          color: 'text.secondary',
          fontSize: '0.875rem',
        }}>
        Map fields from your calls to the corresponding dataset columns.
      </Typography>
      <Box
        sx={{
          bgcolor: '#F8F8F8',
          border: '1px solid #E0E0E0',
          p: 2,
          borderRadius: 1,
        }}>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: '1fr 40px 1fr',
            gap: 2,
            mb: 2,
          }}>
          <Typography
            component="div"
            sx={{
              ...typographyStyle,
              color: 'text.secondary',
              fontSize: '0.875rem',
              pl: 1,
            }}>
            Call Fields
          </Typography>
          <Box /> {/* Spacer for arrow */}
          <Typography
            component="div"
            sx={{
              ...typographyStyle,
              color: 'text.secondary',
              fontSize: '0.875rem',
            }}>
            Dataset Columns
          </Typography>
        </Box>
        <Stack spacing={2}>
          {targetSchema.map(targetField => {
            const currentMapping = fieldMappings.find(
              m => m.targetField === targetField.name
            );

            return (
              <Box
                key={targetField.name}
                sx={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 40px 1fr',
                  alignItems: 'center',
                  gap: 2,
                }}>
                <Box sx={{flex: 1, minWidth: '100px'}}>
                  <Box sx={{position: 'relative'}}>
                    {currentMapping && (
                      <DataPreviewTooltip
                        rows={fieldPreviews.get(currentMapping.sourceField)}
                        tooltipProps={{
                          placement: 'right-start',
                          componentsProps: {
                            popper: {
                              modifiers: [
                                {
                                  name: 'offset',
                                  options: {
                                    offset: [0, 2],
                                  },
                                },
                              ],
                            },
                          },
                        }}>
                        <Box
                          sx={{
                            position: 'absolute',
                            inset: 0,
                            zIndex: 1,
                            pointerEvents: 'none',
                          }}
                        />
                      </DataPreviewTooltip>
                    )}
                    <Select
                      placeholder="Select column"
                      value={
                        currentMapping
                          ? {
                              label: currentMapping.sourceField,
                              value: currentMapping.sourceField,
                            }
                          : null
                      }
                      options={[
                        {label: '-- None --', value: ''},
                        ...sourceSchema.map(field => ({
                          label: field.name,
                          value: field.name,
                        })),
                      ]}
                      onChange={option => {
                        handleMappingChange(
                          targetField.name,
                          option === null ? null : option.value
                        );
                      }}
                      formatOptionLabel={formatOptionLabel}
                      isSearchable={true}
                      isClearable={true}
                    />
                  </Box>
                </Box>
                <Box sx={{display: 'flex', justifyContent: 'center'}}>
                  <Icon name="forward-next" />
                </Box>
                <Box sx={{flex: 1, minWidth: '100px'}}>
                  {renderTargetField(targetField.name)}
                </Box>
              </Box>
            );
          })}
        </Stack>
      </Box>
    </Stack>
  );
};
