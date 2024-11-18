import { createChatCompletion } from '../../../../ai/config';
import { filterGPTMessageByMaxTokens, loadRequestMessages } from '../../../../chat/utils';
import {
  ChatCompletion,
  StreamChatType,
  ChatCompletionMessageParam,
  ChatCompletionAssistantMessageParam
} from '@fastgpt/global/core/ai/type';
import { NextApiResponse } from 'next';
import { responseWriteController } from '../../../../../common/response';
import { SseResponseEventEnum } from '@fastgpt/global/core/workflow/runtime/constants';
import { textAdaptGptResponse } from '@fastgpt/global/core/workflow/runtime/utils';
import { ChatCompletionRequestMessageRoleEnum } from '@fastgpt/global/core/ai/constants';
import { dispatchWorkFlow } from '../../index';
import { DispatchToolModuleProps, RunToolResponse, ToolNodeItemType } from './type.d';
import json5 from 'json5';
import { countGptMessagesTokens } from '../../../../../common/string/tiktoken/index';
import {
  getNanoid,
  replaceVariable,
  sliceJsonStr,
  sliceStrStartEnd
} from '@fastgpt/global/common/string/tools';
import { AIChatItemType } from '@fastgpt/global/core/chat/type';
import { GPTMessages2Chats } from '@fastgpt/global/core/chat/adapt';
import { formatToolResponse, initToolCallEdges, initToolNodes } from './utils';
import { computedMaxToken, llmCompletionsBodyFormat } from '../../../../ai/utils';
import { WorkflowResponseType } from '../../type';
import { toolValueTypeList } from '@fastgpt/global/core/workflow/constants';
import { WorkflowInteractiveResponseType } from '@fastgpt/global/core/workflow/template/system/interactive/type';
import { ChatItemValueTypeEnum } from '@fastgpt/global/core/chat/constants';
import { i18nT } from '../../../../../../web/i18n/utils';
import { addLog } from '../../../../../common/system/log/';

type FunctionCallCompletion = {
  id: string;
  name: string;
  arguments: string;
  toolName?: string;
  toolAvatar?: string;
};

const ERROR_TEXT = 'Tool run error';
const INTERACTIVE_STOP_SIGNAL = 'INTERACTIVE_STOP_SIGNAL';

export const runToolWithPromptCall = async (
  props: DispatchToolModuleProps,
  response?: RunToolResponse
): Promise<RunToolResponse> => {
  const { messages, toolNodes, toolModel, interactiveEntryToolParams, ...workflowProps } = props;
  const {
    res,
    requestOrigin,
    runtimeNodes,
    runtimeEdges,
    user,
    stream,
    workflowStreamResponse,
    params: { temperature = 0, maxToken = 4000, aiChatVision }
  } = workflowProps;

  if (interactiveEntryToolParams) {
    addLog.debug(`交互式工具调用...`);
    initToolNodes(runtimeNodes, interactiveEntryToolParams.entryNodeIds);
    initToolCallEdges(runtimeEdges, interactiveEntryToolParams.entryNodeIds);

    // Run entry tool
    const toolRunResponse = await dispatchWorkFlow({
      ...workflowProps,
      isToolCall: true
    });
    const stringToolResponse = formatToolResponse(toolRunResponse.toolResponses);

    workflowStreamResponse?.({
      event: SseResponseEventEnum.toolResponse,
      data: {
        tool: {
          id: interactiveEntryToolParams.toolCallId,
          toolName: '',
          toolAvatar: '',
          params: '',
          response: sliceStrStartEnd(stringToolResponse, 5000, 5000)
        }
      }
    });

    // Check interactive response(Only 1 interaction is reserved)
    const workflowInteractiveResponseItem = toolRunResponse?.workflowInteractiveResponse
      ? toolRunResponse
      : undefined;

    // Rewrite toolCall messages
    const concatMessages = [...messages.slice(0, -1), ...interactiveEntryToolParams.memoryMessages];
    const lastMessage = concatMessages[concatMessages.length - 1];
    addLog.debug(`LLMRequest: ${lastMessage.content}`);
    lastMessage.content = workflowInteractiveResponseItem
      ? lastMessage.content
      : replaceVariable(lastMessage.content, {
          [INTERACTIVE_STOP_SIGNAL]: stringToolResponse
        });

    // Check stop signal
    const hasStopSignal = toolRunResponse.flowResponses.some((item) => !!item.toolStop);
    if (hasStopSignal || workflowInteractiveResponseItem) {
      // Get interactive tool data
      const workflowInteractiveResponse =
        workflowInteractiveResponseItem?.workflowInteractiveResponse;
      const toolWorkflowInteractiveResponse: WorkflowInteractiveResponseType | undefined =
        workflowInteractiveResponse
          ? {
              ...workflowInteractiveResponse,
              toolParams: {
                entryNodeIds: workflowInteractiveResponse.entryNodeIds,
                toolCallId: '',
                memoryMessages: [lastMessage]
              }
            }
          : undefined;

      return {
        dispatchFlowResponse: [toolRunResponse],
        toolNodeTokens: 0,
        completeMessages: concatMessages,
        assistantResponses: toolRunResponse.assistantResponses,
        runTimes: toolRunResponse.runTimes,
        toolWorkflowInteractiveResponse
      };
    }

    addLog.debug(`return runToolWithPromptCall again, 1`);
    return runToolWithPromptCall(
      {
        ...props,
        interactiveEntryToolParams: undefined,
        messages: concatMessages
      },
      {
        dispatchFlowResponse: [toolRunResponse],
        toolNodeTokens: 0,
        assistantResponses: toolRunResponse.assistantResponses,
        runTimes: toolRunResponse.runTimes
      }
    );
  }

  const assistantResponses = response?.assistantResponses || [];

  const toolsPrompt = JSON.stringify(
    toolNodes.map((item) => {
      const properties: Record<
        string,
        {
          type: string;
          description: string;
          required?: boolean;
          enum?: string[];
        }
      > = {};
      item.toolParams.forEach((item) => {
        const jsonSchema = (
          toolValueTypeList.find((type) => type.value === item.valueType) || toolValueTypeList[0]
        ).jsonSchema;

        properties[item.key] = {
          ...jsonSchema,
          description: item.toolDescription || '',
          enum: item.enum?.split('\n').filter(Boolean) || []
        };
      });

      return {
        toolId: item.nodeId,
        description: item.intro,
        parameters: {
          type: 'object',
          properties,
          required: item.toolParams.filter((item) => item.required).map((item) => item.key)
        }
      };
    })
  );

  const lastMessage = messages[messages.length - 1];
  addLog.debug(`LLMRequest: ${lastMessage.content}`);
  if (typeof lastMessage.content === 'string') {
    lastMessage.content = replaceVariable(lastMessage.content, {
      toolsPrompt
    });
  } else if (Array.isArray(lastMessage.content)) {
    // array, replace last element
    const lastText = lastMessage.content[lastMessage.content.length - 1];
    if (lastText.type === 'text') {
      lastText.text = replaceVariable(lastText.text, {
        toolsPrompt
      });
    } else {
      return Promise.reject('Prompt call invalid input');
    }
  } else {
    return Promise.reject('Prompt call invalid input');
  }

  const filterMessages = await filterGPTMessageByMaxTokens({
    messages,
    maxTokens: toolModel.maxContext - 500 // filter token. not response maxToken
  });

  const [requestMessages, max_tokens] = await Promise.all([
    loadRequestMessages({
      messages: filterMessages,
      useVision: toolModel.vision && aiChatVision,
      origin: requestOrigin
    }),
    computedMaxToken({
      model: toolModel,
      maxToken,
      filterMessages
    })
  ]);
  const requestBody = llmCompletionsBodyFormat(
    {
      model: toolModel.model,
      temperature,
      max_tokens,
      stream,
      messages: requestMessages
    },
    toolModel
  );

  // console.log(JSON.stringify(requestMessages, null, 2));
  /* Run llm */
  const { response: aiResponse, isStreamResponse } = await createChatCompletion({
    body: requestBody,
    userKey: user.openaiAccount,
    options: {
      headers: {
        Accept: 'application/json, text/plain, */*'
      }
    }
  });

  const answer = await (async () => {
    if (res && isStreamResponse) {
      const { answer } = await streamResponse({
        res,
        toolNodes,
        stream: aiResponse,
        workflowStreamResponse
      });

      return answer;
    } else {
      const result = aiResponse as ChatCompletion;

      return result.choices?.[0]?.message?.content || '';
    }
  })();

  const { answer: replaceAnswer, toolJson, msg } = parseAnswer(answer);
  // No tools
  if (!toolJson) {
    addLog.log(`Not toolJson, FinalResponse: ${replaceAnswer}`);
    if (replaceAnswer === ERROR_TEXT && msg) {
      workflowStreamResponse?.({
        event: SseResponseEventEnum.answer,
        data: textAdaptGptResponse({
          text: replaceAnswer
        })
      });
    }

    // 不支持 stream 模式的模型的流式响应
    if (stream && !isStreamResponse) {
      workflowStreamResponse?.({
        event: SseResponseEventEnum.fastAnswer,
        data: textAdaptGptResponse({
          text: replaceAnswer
        })
      });
    }

    // No tool is invoked, indicating that the process is over
    const gptAssistantResponse: ChatCompletionAssistantMessageParam = {
      role: ChatCompletionRequestMessageRoleEnum.Assistant,
      content: replaceAnswer
    };
    const completeMessages = filterMessages.concat(gptAssistantResponse);
    const tokens = await countGptMessagesTokens(completeMessages, undefined);
    // console.log(tokens, 'response token');

    // concat tool assistant
    const toolNodeAssistant = GPTMessages2Chats([gptAssistantResponse])[0] as AIChatItemType;

    return {
      dispatchFlowResponse: response?.dispatchFlowResponse || [],
      toolNodeTokens: response?.toolNodeTokens ? response.toolNodeTokens + tokens : tokens,
      completeMessages,
      assistantResponses: [...assistantResponses, ...toolNodeAssistant.value],
      runTimes: (response?.runTimes || 0) + 1
    };
  }

  // Run the selected tool.
  const toolsRunResponse = await (async () => {
    const toolNode = toolNodes.find((item) => item.nodeId === toolJson.name);
    if (!toolNode) return Promise.reject('tool not found');

    toolJson.toolName = toolNode.name;
    toolJson.toolAvatar = toolNode.avatar;

    // run tool flow
    const startParams = (() => {
      try {
        return json5.parse(toolJson.arguments);
        addLog.log(`ToolCall, Arguments parsed successfully: `, parsedArguments);
      } catch (error) {
        addLog.error(`ToolCall, Failed to parse arguments: `, (error as Error).message);
        return {};
      }
    })();

    // SSE response to client
    workflowStreamResponse?.({
      event: SseResponseEventEnum.toolCall,
      data: {
        tool: {
          id: toolJson.id,
          toolName: toolNode.name,
          toolAvatar: toolNode.avatar,
          functionName: toolJson.name,
          params: toolJson.arguments,
          response: ''
        }
      }
    });

    addLog.debug('非交互式工具调用...');
    initToolNodes(runtimeNodes, [toolNode.nodeId], startParams);
    const toolResponse = await dispatchWorkFlow({
      ...workflowProps,
      isToolCall: true
    });

    const stringToolResponse = formatToolResponse(toolResponse.toolResponses);
    addLog.log(`Tool call success!`);

    workflowStreamResponse?.({
      event: SseResponseEventEnum.toolResponse,
      data: {
        tool: {
          id: toolJson.id,
          toolName: '',
          toolAvatar: '',
          params: '',
          response: sliceStrStartEnd(stringToolResponse, 500, 500)
        }
      }
    });

    return {
      toolResponse,
      toolResponsePrompt: stringToolResponse
    };
  })();

  // 合并工具调用的结果，使用 functionCall 格式存储。
  const assistantToolMsgParams: ChatCompletionAssistantMessageParam = {
    role: ChatCompletionRequestMessageRoleEnum.Assistant,
    function_call: toolJson
  };

  /* 
    ...
    user
    assistant: tool data
  */
  const concatToolMessages = [
    ...requestMessages,
    assistantToolMsgParams
  ] as ChatCompletionMessageParam[];
  // Only toolCall tokens are counted here, Tool response tokens count towards the next reply
  const tokens = await countGptMessagesTokens(concatToolMessages, undefined);

  /* 
    ...
    user
    assistant: tool data
    function: tool response
  */
  const functionResponseMessage: ChatCompletionMessageParam = {
    role: ChatCompletionRequestMessageRoleEnum.Function,
    name: toolJson.name,
    content: toolsRunResponse.toolResponsePrompt
  };

  // tool node assistant
  const toolNodeAssistant = GPTMessages2Chats([
    assistantToolMsgParams,
    functionResponseMessage
  ])[0] as AIChatItemType;
  const toolChildAssistants = toolsRunResponse.toolResponse.assistantResponses.filter(
    (item) => item.type !== ChatItemValueTypeEnum.interactive
  );
  const toolNodeAssistants = [
    ...assistantResponses,
    ...toolNodeAssistant.value,
    ...toolChildAssistants
  ];

  const dispatchFlowResponse = response
    ? [...response.dispatchFlowResponse, toolsRunResponse.toolResponse]
    : [toolsRunResponse.toolResponse];

  // Check interactive response(Only 1 interaction is reserved)
  const workflowInteractiveResponseItem = toolsRunResponse.toolResponse?.workflowInteractiveResponse
    ? toolsRunResponse.toolResponse
    : undefined;

  // get the next user prompt
  if (typeof lastMessage.content === 'string') {
    lastMessage.content += `${replaceAnswer}
TOOL_RESPONSE: """
${workflowInteractiveResponseItem ? `{{${INTERACTIVE_STOP_SIGNAL}}}` : toolsRunResponse.toolResponsePrompt}
"""
ANSWER: `;
  } else if (Array.isArray(lastMessage.content)) {
    // array, replace last element
    const lastText = lastMessage.content[lastMessage.content.length - 1];
    if (lastText.type === 'text') {
      lastText.text += `${replaceAnswer}
TOOL_RESPONSE: """
${workflowInteractiveResponseItem ? `{{${INTERACTIVE_STOP_SIGNAL}}}` : toolsRunResponse.toolResponsePrompt}
"""
ANSWER: `;
    } else {
      return Promise.reject('Prompt call invalid input');
    }
  } else {
    return Promise.reject('Prompt call invalid input');
  }

  const runTimes = (response?.runTimes || 0) + toolsRunResponse.toolResponse.runTimes;
  const toolNodeTokens = response?.toolNodeTokens ? response.toolNodeTokens + tokens : tokens;
  addLog.log(`Tool runTimes: ${runTimes}`);

  // Check stop signal
  const hasStopSignal = toolsRunResponse.toolResponse.flowResponses.some((item) => !!item.toolStop);

  if (hasStopSignal || workflowInteractiveResponseItem) {
    // Get interactive tool data
    const workflowInteractiveResponse =
      workflowInteractiveResponseItem?.workflowInteractiveResponse;
    const toolWorkflowInteractiveResponse: WorkflowInteractiveResponseType | undefined =
      workflowInteractiveResponse
        ? {
            ...workflowInteractiveResponse,
            toolParams: {
              entryNodeIds: workflowInteractiveResponse.entryNodeIds,
              toolCallId: '',
              memoryMessages: [lastMessage]
            }
          }
        : undefined;

    return {
      dispatchFlowResponse,
      toolNodeTokens,
      completeMessages: filterMessages,
      assistantResponses: toolNodeAssistants,
      runTimes,
      toolWorkflowInteractiveResponse
    };
  }

  addLog.debug(`return runToolWithPromptCall again: 2`);
  return runToolWithPromptCall(
    {
      ...props,
      messages
    },
    {
      dispatchFlowResponse,
      toolNodeTokens,
      assistantResponses: toolNodeAssistants,
      runTimes
    }
  );
};

async function streamResponse({
  res,
  stream,
  workflowStreamResponse
}: {
  res: NextApiResponse;
  toolNodes: ToolNodeItemType[];
  stream: StreamChatType;
  workflowStreamResponse?: WorkflowResponseType;
}) {
  const write = responseWriteController({
    res,
    readStream: stream
  });

  let startResponseWrite = false;
  let textAnswer = '';
  let text = '';
  const prefixReg = /^1(:|：|,|，)/;
  const answerPrefixReg = /^0(:|：|,|，)/;

  for await (const part of stream) {
    if (res.closed) {
      addLog.warn(`Response closed, aborting stream...`);
      stream.controller?.abort();
      break;
    }

    const responseChoice = part.choices?.[0]?.delta;
    addLog.log(`responseChoice:`, responseChoice);

    if (responseChoice?.content) {
      const content = responseChoice?.content || '';
      textAnswer += content;

      if (startResponseWrite) {
        workflowStreamResponse?.({
          write,
          event: SseResponseEventEnum.answer,
          data: textAdaptGptResponse({
            text: content
          })
        });
      } else if (textAnswer.length >= 3) {
        textAnswer = textAnswer.trim();
        // 如果还没开始写入响应，检查前缀
        if (!prefixReg.test(textAnswer) && !answerPrefixReg.test(textAnswer)) {
          // 如果不符合条件，则将其改为以 '0: ' 开头
          textAnswer = '0: ' + textAnswer;
        }
        if (answerPrefixReg.test(textAnswer)) {
          startResponseWrite = true;
          // find first : index
          text = textAnswer.replace(answerPrefixReg, '').trim();
          workflowStreamResponse?.({
            write,
            event: SseResponseEventEnum.answer,
            data: textAdaptGptResponse({
              text: text
            })
          });
        }
      }
    }
  }

  if (!textAnswer) {
    return Promise.reject(i18nT('chat:LLM_model_response_empty'));
  }
  return { answer: textAnswer.trim() };
}

const parseAnswer = (
  str: string
): {
  answer: string;
  toolJson?: FunctionCallCompletion;
  msg?: string; // 添加 msg
} => {
  str = str.trim();
  addLog.debug(`LLMResponse: ${str}`); // 打印 str 的值
  const prefixReg = /^1(:|：|,|，)/;
  const answerPrefixReg = /^0(:|：|,|，)/;

  // 首先，使用正则表达式提取TOOL_ID和TOOL_ARGUMENTS
  if (prefixReg.test(str)) {
    str = str.replace(prefixReg, '1:').trim();
    const toolString = sliceJsonStr(str);

    try {
      const toolCall = json5.parse(toolString);
      addLog.debug(`ToolCall: `, toolCall);
      return {
        answer: `1: ${toolString}`,
        toolJson: {
          id: getNanoid(),
          name: toolCall.toolId,
          arguments: JSON.stringify(toolCall.arguments || toolCall.parameters)
        }
      };
    } catch (error) {
      addLog.error(`ToolCallError: ${str}`);
      return {
        answer: str,
        msg: ERROR_TEXT // 返回
      };
    }
  } else {
    if (answerPrefixReg.test(str)) {
      str = str.replace(answerPrefixReg, '').trim();
    }
    return {
      answer: str
    };
  }
};
