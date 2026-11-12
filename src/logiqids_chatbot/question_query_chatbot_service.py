import json

from django.conf import settings

from common.v1.exceptions import NotAcceptableError
from common.v1.services.admin_config_service import AdminConfigService
from common.v1.services.tracking_details_service import TrackingDetailsService
from common.v1.utils.date_format import Date
from content_package.models import QueryChatSpamResponses, QueryGPTResponseBank, QuestionQueries, QuestionQueriesChats
from content_package.v1.constants.chatbot_constants import (
    CHATBOT_ATTEMPT_1_INSTRUCTIONS,
    CHATBOT_ATTEMPT_2_EXAMPLES_USED,
    CHATBOT_ATTEMPT_2_FORMULA_BASED,
    CHATBOT_ATTEMPT_2_GENERAL,
    CHATBOT_ATTEMPT_2_PREFIX,
    CHATBOT_ATTEMPT_2_SUFFIX,
    CHATBOT_INSTRUCTION_BREAKDOWN_QUESTION,
    CHATBOT_INSTRUCTION_DEFEND_CORRECT_ANSWER,
    CHATBOT_INSTRUCTION_SIMPLIFY_SOLUTION,
    CHATBOT_RAG_INSTRUCTIONS_PREFIX,
    OTHER_DOUBT_RECLASSIFICATION_ENABLED,
    RAG_HIGH_RELEVANCE_THRESHOLD,
)
from content_package.v1.constants.constants import (
    CHATBOT_JSON_SYSTEM_MESSAGE,
    PAPER_TYPE,
    QUESTION_QUERY_PROMPT_DETAILS,
    QUESTION_QUERY_PROMPT_IDETIFIER_MAPPING,
    STORED_RESPONSE_IDENTIFIERS,
)
from content_package.v1.services.chatbot.other_doubt_reclassification_service import (
    RECLASSIFIED_ADMIN_IDENTIFIERS,
    RECLASSIFIED_IDENTIFIER_TO_INSTRUCTIONS,
    OtherDoubtReclassificationService,
)
from content_package.v1.services.chatbot.query_bot_logging_service import QueryBotLoggingService
from content_package.v1.services.chatbot.query_classification_service import QueryClassificationService
from content_package.v1.services.chatbot.query_types import QueryContext, QueryType
from content_package.v1.services.chatbot.rag.question_embedding_service import QuestionEmbeddingService
from content_package.v1.services.chatbot.rag.rag_service import RAGService
from content_package.v1.services.chatbot.resolve_keywords_matcher import ResolveKeywordsMatcher
from content_package.v1.services.question_query_automation_service import QueryAutomation
from content_package.v2.services.question_query_service import QuestionDetailsService, QuestionQueryService
from user.v1.services.user_info_service import UserInfoService


class QuestionQueryChatBotService:
    def __init__(self, question_id=None, query_id=None, user_id=None, source="question"):
        """Initialise an instance of the service."""
        self.user_id = user_id
        self.question_id = question_id
        if query_id:
            self.set_query_id(query_id)
        self.typing_enabled = False
        self.profile_url = UserInfoService(user_id=user_id).get_profile_img(True)
        self.source = source

    def set_query_id(self, query_id):
        query_obj = QuestionQueries.objects.filter(user_id=self.user_id, id=query_id).last()
        if not query_obj:
            raise NotAcceptableError("Invalid query id")
        self.query_obj = query_obj
        self.question_id = query_obj.question_id

    # query initialization

    def initialize_chat(self, paper_id=None, paper_type=None):
        if paper_id and paper_type:
            PAPER_TYPE_DICT = dict((v, k) for k, v in PAPER_TYPE.items())
            paper_type_value = PAPER_TYPE_DICT.get(paper_type)
            # app paper type bug handling
            if paper_type_value == 1:
                from content_package.models import MistakeWorksheets, Worksheets

                if not Worksheets.objects.filter(id=paper_id, user_id=self.user_id).exists():
                    if MistakeWorksheets.objects.filter(id=paper_id, user_id=self.user_id).exists():
                        paper_type_value = 5
            self.paper_type = paper_type_value
            query_obj = QuestionQueries.objects.filter(
                question_id=self.question_id, paper_id=paper_id, paper_type=paper_type_value, user_id=self.user_id
            ).last()
            if query_obj:
                self.query_obj = query_obj
        return self.get_user_chat_resonse()

    def get_user_chat_resonse(self):
        self.set_typing_enabled("get")
        response = {
            "chat_history": self.get_chat_history(),
            "query_resolved": self.get_query_resolved(),
            "question_details": self.get_question_details(),
            "typing_enabled": self.typing_enabled,
            "is_bot": self.get_is_bot(),
        }
        if hasattr(self, "query_obj"):
            response["query_id"] = self.query_obj.id
            self.user_read_messages()
        return response

    def get_question_details(self):
        if hasattr(self, "query_obj"):
            return QuestionQueryService(self.query_obj.id).get_question_details()
        return QuestionDetailsService(self.question_id, self.get_paper_type()).get_question_details()

    def get_query_resolved(self):
        if hasattr(self, "query_obj"):
            return self.query_obj.is_resolved
        return False

    def get_is_bot(self):
        if hasattr(self, "query_obj"):
            return self.query_obj.chatbot
        return True

    def get_paper_type(self):
        if hasattr(self, "query_obj"):
            return self.query_obj.paper_type
        if hasattr(self, "paper_type"):
            return self.paper_type
        return None

    def get_chat_history(self):
        history = []
        if hasattr(self, "query_obj"):
            if self.query_obj.identifier:
                system_message_obj = self.get_system_message(
                    selection_enabled=False, date_obj=Date.get_datetime_from_unix_time(self.query_obj.created_at)
                )
                history.append(system_message_obj)
            history.append(self.get_user_query())
            question_queriy_chats = QuestionQueriesChats.objects.filter(
                question_query_id=self.query_obj.id, is_active=True
            ).order_by("created_at")
            total_chat_length = len(question_queriy_chats)
            if total_chat_length > 0:
                for index, question_queriy_chat_obj in enumerate(question_queriy_chats):
                    if question_queriy_chat_obj.type == "selection":
                        system_message_obj = self.get_system_message(
                            identifier=self.query_obj.identifier,
                            selection_enabled=False,
                            date_obj=question_queriy_chat_obj.created_at,
                        )
                        history.append(system_message_obj)
                    history.append(
                        self.get_user_query(
                            question_queriy_chat_obj, selection_enabled=((index + 1) == total_chat_length)
                        )
                    )
            elif self.query_obj.identifier:
                history.append(self.get_system_message(identifier=self.query_obj.identifier))
        else:
            self.set_query_chatbot_value()
            history.append(self.get_system_message(identifier=None if self.QUERY_CHATBOT else "chatbot_disabled"))
        return self.format_chat_history(history)

    def get_profile_url(self, chat_from):
        if chat_from.upper() == "USER":
            return self.profile_url
        return "https://lqcdndata.s3.ap-south-1.amazonaws.com/chat-icon.png"

    def get_user_query(self, chat_obj=None, selection_enabled=True):
        if chat_obj:
            results = {
                "from": chat_obj.chat_from,
                "text": chat_obj.chat_text,
                "image": chat_obj.chat_image,
                "profile_url": self.get_profile_url(chat_obj.chat_from),
                "response_feedback": self.get_response_feedback_obj(chat_obj.type),
                "selection_enabled": selection_enabled,
            }
            results.update(self.get_date_time_formatted(date_obj=chat_obj.created_at))
        else:
            results = {
                "from": "USER",
                "text": self.query_obj.query,
                "image": self.query_obj.screenshot_link,
                "profile_url": self.get_profile_url("USER"),
            }
            results.update(self.get_date_time_formatted(unix_date_obj=self.query_obj.created_at))
        return results

    def format_chat_history(self, history):
        date_wise_history = {}
        for chat_obj in history:
            if not date_wise_history.get(chat_obj["date"]):
                date_wise_history[chat_obj["date"]] = []
            date_wise_history[chat_obj["date"]].append(chat_obj)
        results = [
            {"date": date_history_obj[0], "data": date_history_obj[1]} for date_history_obj in date_wise_history.items()
        ]
        return results

    def get_system_message(self, date_obj=None, selection_enabled=True, identifier=None):
        if not date_obj:
            date_obj = Date.get_current_local_datetime()
        text = ""
        response_prompt = []
        if identifier == "chatbot_disabled":
            text = "Hi! Please explain the problem you are facing in detail."
        elif identifier and QUESTION_QUERY_PROMPT_DETAILS[identifier].get("response_prompt"):
            response_prompt = QUESTION_QUERY_PROMPT_DETAILS[identifier].get("response_prompt")
            text = "Tell us more"
        else:
            text = "Hi 👋 I am your virtual assistant for doubt resolution. How can I help you?"
            response_prompt = []
            for identifier in QUESTION_QUERY_PROMPT_DETAILS.values():
                if identifier.get("validation_rule"):
                    if not getattr(self, identifier.get("validation_rule"))():
                        continue
                response_prompt.append(identifier.get("text"))

        results = {
            "from": "BOT",
            "text": text,
            "image": None,
            "selection_enabled": selection_enabled,
            "response_prompt": response_prompt,
            "response_feedback": None,
            "profile_url": self.get_profile_url("BOT"),
        }
        results.update(self.get_date_time_formatted(date_obj=date_obj))
        return results

    def get_date_time_formatted(self, unix_date_obj=False, date_obj=False):
        if unix_date_obj:
            date_obj = Date.get_datetime_from_unix_time(unix_date_obj)
        if date_obj.tzinfo != settings.TIME_ZONE:
            date_obj = Date.convert_to_timezone(date_obj, settings.TIME_ZONE)
        return {"date": Date.to_short_string(date_obj), "time": Date.to_time_string(date_obj)}

    def get_response_feedback_obj(self, type):
        if type not in ["solution", "try_again"]:
            return None
        options = ["Yes", "No"]
        question = "Was this helpful?"
        if type == "try_again":
            question = "Would you like to try again?"
        return {"question": question, "options": options}

    # save query user response

    def create_query_and_generate_response(self, data, auth_data):
        text = data.get("text")
        image = data.get("image")
        paper_id = data.get("paper_id")
        PAPER_TYPE_DICT = dict((v, k) for k, v in PAPER_TYPE.items())
        paper_type_text = data.get("paper_type")
        paper_type = PAPER_TYPE_DICT.get(paper_type_text)
        if not paper_type:
            raise NotAcceptableError("Invalid Paper Type")
        # app paper type bug handling
        from common.v1.utils.general_utility_functions import set_app_source_type

        source = auth_data.get("source")
        source = set_app_source_type(source)
        if source != "web" and paper_type == 1:
            from content_package.models import MistakeWorksheets, Worksheets

            if not Worksheets.objects.filter(id=paper_id, user_id=self.user_id).exists():
                if MistakeWorksheets.objects.filter(id=paper_id, user_id=self.user_id).exists():
                    paper_type = 5
        TrackingDetailsService(
            model_id=self.question_id, model_name=f"question_query ({paper_type_text.lower()})"
        ).add_details(**auth_data)
        unix_date_now = Date.current_unix_timestamp()
        self.set_query_chatbot_value()
        if self.QUERY_CHATBOT:
            identifier = QUESTION_QUERY_PROMPT_IDETIFIER_MAPPING.get(text)
            if not identifier:
                raise NotAcceptableError("Invalid response")
            query_type = QUESTION_QUERY_PROMPT_DETAILS[identifier]["type"]
            query_status = 0
        else:
            identifier = "chatbot_disabled"
            query_type = 3
            query_status = 1
        query_obj = QuestionQueries.objects.filter(
            question_id=self.question_id, paper_id=paper_id, paper_type=paper_type
        ).last()
        if query_obj:
            self.query_obj = query_obj
            raise NotAcceptableError("Query already generated")
        else:
            self.query_obj = QuestionQueries.objects.create(
                question_id=self.question_id,
                query=text,
                query_type=query_type,
                user_id=self.user_id,
                paper_id=paper_id,
                paper_type=paper_type,
                created_at=unix_date_now,
                updated_at=unix_date_now,
                created_by=self.user_id,
                updated_by=self.user_id,
                status=query_status,
                screenshot_link=image,
                chatbot=True,
                identifier=identifier,
            )
            self.set_user_message(text=text, image=image, date_obj=Date.now())
        return self.generate_response(identifier)

    def save_and_generate_response(self, data):
        text = data.get("text")
        image = data.get("image")
        chat_type = None
        selection_identifier = None
        self.set_query_chatbot_value()
        if self.query_obj.identifier:
            if self.query_obj.chatbot:
                self.validate_user_response(text)
            previous_chat_type = self.get_previous_chat_type()
            if previous_chat_type in ["solution"]:
                chat_type = "feedback"
            if previous_chat_type in ["try_again"]:
                chat_type = "try_again_feedback"
            if previous_chat_type == "explain":
                chat_type = "explain"
            prompt_details = QUESTION_QUERY_PROMPT_DETAILS.get(self.query_obj.identifier)
            if (
                not previous_chat_type
                and prompt_details
                and prompt_details.get("response_prompt")
                and self.query_obj.chatbot
            ):
                chat_type = "selection"
                selection_identifier = QUESTION_QUERY_PROMPT_IDETIFIER_MAPPING.get(text)
                if not selection_identifier:
                    raise NotAcceptableError("Invalid selection")
            if image and not text:
                chat_type = "only_img"
            if previous_chat_type == "feedback" and not self.query_obj.chatbot:
                chat_type = "explain"
        self.add_chat(
            chat_from="USER",
            chat_text=text,
            chat_image=image,
            type=chat_type,
            selection_identifier=selection_identifier,
        )
        self.set_user_message(text=text, image=image, date_obj=Date.now())
        if chat_type == "only_img":
            return self.handle_only_image()
        if chat_type in ["feedback", "explain"] and ResolveKeywordsMatcher(
            only_exact_match=self.query_obj.chatbot
        ).should_resolve(text):
            self.on_feedback("yes")
            self.resolve_query()
        return self.generate_response(identifier=self.get_identifier())

    def handle_only_image(self):
        self.typing_enabled = True
        only_image_count = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, is_active=True, chat_from="USER", type="only_img"
        ).count()
        if only_image_count > 1:
            reply_text = "We're not sure what you're asking about in the image. Can you please type your doubt and mention which part of the image you're referring to?"
        else:
            reply_text = "Can you please briefly explain the problem you are facing"
        chat_obj = self.add_chat(chat_from="BOT", chat_text=reply_text, type="img_desc")
        return self.get_response_structure(chat_obj=chat_obj)

    def get_identifier(self):
        question_query_chat_obj = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, is_active=True, type="selection"
        ).last()
        if question_query_chat_obj:
            if question_query_chat_obj.selection_identifier:
                return question_query_chat_obj.selection_identifier
        return self.query_obj.identifier

    def validate_user_response(self, text):
        question_query_chat_obj = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, is_active=True
        ).last()
        if not question_query_chat_obj:
            return True
        if question_query_chat_obj.chat_from in ["USER"]:
            raise NotAcceptableError("Please wait for the response")
        if question_query_chat_obj.chat_from in ["BOT", "ADMIN"]:
            if question_query_chat_obj.type in ["solution", "try_again"]:
                if text.lower() not in ["yes", "no"]:
                    raise NotAcceptableError("Response should be in Yes or No")
        return True

    # query bot responses

    def generate_response(self, identifier):
        chat_obj = None
        if self.query_obj.is_resolved:
            if QUESTION_QUERY_PROMPT_DETAILS.get(self.query_obj.identifier):
                resolved_text = QUESTION_QUERY_PROMPT_DETAILS[self.query_obj.identifier].get("resolved_text")
                if resolved_text:
                    chat_obj = self.add_chat(chat_from="BOT", chat_text=resolved_text, type="resolved")
        elif self.query_obj.chatbot:
            function_name = f"get_{identifier}_response"
            if hasattr(self, function_name):
                chat_obj = getattr(self, function_name)(identifier)
            elif QUESTION_QUERY_PROMPT_DETAILS.get(identifier):
                response_prompt = QUESTION_QUERY_PROMPT_DETAILS[identifier].get("response_prompt")
                if response_prompt:
                    message_obj = self.get_system_message(identifier=identifier)
                    return self.get_response_structure(prompt_response=message_obj)
        else:
            self.active_admin_query()
        return self.get_response_structure(chat_obj=chat_obj)

    # def get_could_not_understand_response(self, identifier):
    # if self.bot_solution_count() < 2:
    #     gpt_solution = self.get_chat_gpt_response()
    #     chat_obj = self.add_chat(chat_from="BOT", chat_text=gpt_solution, type="solution")
    # else:
    #     reply_text = QUESTION_QUERY_PROMPT_DETAILS[identifier]["admin_text"]
    #     chat_obj = self.add_chat(chat_from="BOT", chat_text=reply_text, type="noted")
    #     self.query_obj.chatbot = False
    #     self.query_obj.save()
    # self.typing_enabled = False
    # return chat_obj
    # return None

    def get_could_not_understand_question_response(self, identifier):
        solution_count = self.bot_solution_count()
        if solution_count < 2:
            chat_obj = self.get_chat_gpt_response(identifier, instructions=CHATBOT_INSTRUCTION_BREAKDOWN_QUESTION)
            self.typing_enabled = False
        else:
            chat_obj = self.user_chat_eloboratation_flow(identifier)
        return chat_obj

    def get_could_not_understand_solution_response(self, identifier):
        if self.bot_solution_count() < 2:
            chat_obj = self.get_chat_gpt_response(identifier, instructions=CHATBOT_INSTRUCTION_SIMPLIFY_SOLUTION)
            self.typing_enabled = False
        else:
            chat_obj = self.user_chat_eloboratation_flow(identifier)
        return chat_obj

    def get_question_wrong_response(self, identifier):
        return self.user_chat_eloboratation_flow(identifier)

    def get_answer_incorrect_response(self, identifier):
        if self.bot_solution_count() < 1:
            chat_obj = self.get_chat_gpt_response(identifier, instructions=CHATBOT_INSTRUCTION_DEFEND_CORRECT_ANSWER)
        else:
            chat_obj = self.user_chat_eloboratation_flow(identifier)
        return chat_obj

    def get_tech_glitch_response(self, identifier):
        reply_text = QUESTION_QUERY_PROMPT_DETAILS[identifier]["admin_text"]
        chat_obj = self.add_chat(chat_from="BOT", chat_text=reply_text, type="noted")
        self.resolve_query()
        return chat_obj

    def get_typo_response(self, identifier):
        return self.user_chat_eloboratation_flow(identifier)

    def get_other_response(self, identifier):
        return self.user_chat_eloboratation_flow(identifier)

    def get_chatbot_disabled_response(self, identifier=None):
        reply_text = """Thanks for sharing. Someone from our team will reply within 2 working days. Go to the doubts section to check the reply later."""
        return self.send_query_to_admin(reply_text)

    def get_response_structure(self, chat_obj=None, prompt_response=None):
        self.set_typing_enabled("generate")
        response_messages = []
        if hasattr(self, "user_message"):
            response_messages.append(self.user_message)
        if chat_obj:
            chat_response = {
                "from": chat_obj.chat_from,
                "text": chat_obj.chat_text,
                "image": chat_obj.chat_image,
                "response_feedback": self.get_response_feedback_obj(chat_obj.type),
                "profile_url": self.get_profile_url(chat_obj.chat_from),
                "selection_enabled": chat_obj.type in ["solution", "try_again"],
            }
            chat_response.update(self.get_date_time_formatted(date_obj=chat_obj.created_at))
            response_messages.append(chat_response)
        if prompt_response:
            response_messages.append(prompt_response)
        response = {
            "typing_enabled": self.typing_enabled,
            "query_id": self.query_obj.id,
            "query_resolved": self.query_obj.is_resolved,
            "is_bot": self.query_obj.chatbot,
            "response_messages": response_messages,
        }
        return response

    def set_typing_enabled(self, source="get"):
        if hasattr(self, "query_obj"):
            if self.query_obj.is_resolved:
                self.typing_enabled = True
            elif self.query_obj.chatbot:
                if source == "get":
                    last_chat_obj = (
                        QuestionQueriesChats.objects.filter(question_query_id=self.query_obj.id, is_active=True)
                        .exclude(type__in=["only_img", "img_desc"])
                        .last()
                    )
                    if (
                        last_chat_obj
                        and last_chat_obj.type == "explain"
                        and last_chat_obj.chat_from in ["BOT", "ADMIN"]
                    ):
                        self.typing_enabled = True
            elif self.source == "doubts":
                self.typing_enabled = True
        else:
            if not hasattr(self, "QUERY_CHATBOT"):
                self.set_query_chatbot_value()
            if not self.QUERY_CHATBOT:
                self.typing_enabled = True

    def add_chat(self, chat_from, chat_text, chat_image=None, type=None, selection_identifier=None):
        question_query_chat_obj = QuestionQueriesChats.objects.create(
            question_query_id=self.query_obj.id,
            chat_text=chat_text,
            chat_image=chat_image,
            user_id=self.user_id if chat_from == "USER" else None,
            chat_from=chat_from,
            type=type,
            selection_identifier=selection_identifier,
        )
        if question_query_chat_obj.type == "feedback":
            if question_query_chat_obj.chat_text.lower() == "yes":
                self.resolve_query()
            self.on_feedback(question_query_chat_obj.chat_text.lower())
        if question_query_chat_obj.type == "try_again_feedback" and question_query_chat_obj.chat_text.lower() == "no":
            self.resolve_query()
        # if chat_from in ["USER"] and not self.query_obj.chatbot and self.query_obj.status == 0:
        #     self.query_obj.status = 1
        #     self.query_obj.save()
        return question_query_chat_obj

    def resolve_query(self):
        self.query_obj.is_resolved = True
        self.query_obj.status = 0
        self.query_obj.save()

    def send_query_to_admin(self, reply_text):
        chat_obj = self.add_chat(chat_from="BOT", chat_text=reply_text, type="noted")
        self.query_obj.chatbot = False
        self.active_admin_query()
        return chat_obj

    def active_admin_query(self):
        self.query_obj.status = 1
        self.query_obj.save()

    def set_query_chatbot_value(self):
        self.QUERY_CHATBOT = bool(AdminConfigService().get_setting(name="QUERY_CHATBOT", default_value=0))

    def is_rag_enabled(self):
        if not hasattr(self, "is_rag_enabled"):
            self.is_rag_enabled = bool(AdminConfigService().get_setting(name="RAG_ENABLED", default_value=0))
        return self.is_rag_enabled

    def get_previous_chat_type(self, chat_from=None):
        question_query_chat_queryset = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, is_active=True
        ).exclude(type__in=["only_img", "img_desc"])
        if chat_from:
            question_query_chat_queryset = question_query_chat_queryset.filter(chat_from=chat_from)
        question_query_chat_obj = question_query_chat_queryset.last()
        if question_query_chat_obj:
            return question_query_chat_obj.type
        return None

    def bot_solution_count(self):
        return QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, chat_from="BOT", is_active=True, type="solution"
        ).count()

    def get_previous_solution_attempts(self):
        """Get previous solution attempts with their content."""
        previous_solutions = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, chat_from="BOT", is_active=True, type="solution"
        ).order_by("created_at")
        return list(previous_solutions)

    def get_attempt_specific_instructions(self, base_instructions=""):
        """Generate attempt-specific instructions that vary teaching approaches."""
        solution_count = self.bot_solution_count()
        previous_solutions = self.get_previous_solution_attempts()

        if solution_count == 0:
            # Attempt 1: Standard step-by-step explanation
            attempt_instructions = CHATBOT_ATTEMPT_1_INSTRUCTIONS
        elif solution_count == 1:
            # Attempt 2: Significantly different approach
            previous_text = previous_solutions[0].chat_text if previous_solutions else ""
            attempt_instructions = CHATBOT_ATTEMPT_2_PREFIX

            # Analyze previous attempt to suggest different approach
            prev_lower = previous_text.lower()
            if any(word in prev_lower for word in ["formula", "equation", "calculate", "solve"]):
                attempt_instructions += CHATBOT_ATTEMPT_2_FORMULA_BASED
            elif any(word in prev_lower for word in ["example", "instance", "case"]):
                attempt_instructions += CHATBOT_ATTEMPT_2_EXAMPLES_USED
            else:
                attempt_instructions += CHATBOT_ATTEMPT_2_GENERAL

            attempt_instructions += CHATBOT_ATTEMPT_2_SUFFIX
        else:
            # Should not happen (max 2 attempts), but handle gracefully
            attempt_instructions = base_instructions

        if base_instructions:
            return f"{attempt_instructions}\n\n{base_instructions}"
        return attempt_instructions

    # GPT response generation

    def get_chat_gpt_response(
        self,
        identifier=None,
        instructions="",
        system_message="",
        rag_retrieval_result=None,
        routed_identifier=None,
    ):
        instruction_text = self.get_attempt_specific_instructions(instructions)
        chat_history_gpt = []
        query = self.query_obj.query
        chat_history_queryset = QuestionQueriesChats.objects.filter(question_query_id=self.query_obj.id, is_active=True)
        chat_history_gpt.append({"role": "user", "content": query})
        for chat_history_obj in chat_history_queryset:
            role = chat_history_obj.chat_from.lower().replace("bot", "assistant")
            content = chat_history_obj.chat_text
            if chat_history_obj.type == "feedback" and content.lower() == "no":
                content = "This was not helpful. Explain another way."
            chat_history_gpt.append({"role": role, "content": content})
        if identifier:
            IDENTIFIER_MAPPING_DICT = dict((v, k) for k, v in QUESTION_QUERY_PROMPT_IDETIFIER_MAPPING.items())
            query = IDENTIFIER_MAPPING_DICT.get(identifier)
        past_response = self.get_response_from_past_responses(identifier)
        llm_automation = None
        if past_response:
            gpt_response = past_response
        else:
            llm_automation = QueryAutomation(
                query_id=self.query_obj.id,
                query=query,
                instructions=instruction_text,
                chat_history=chat_history_gpt,
                system_message=system_message,
                routed_identifier=routed_identifier,
            )
            gpt_response = llm_automation.generate_response()
        if gpt_response:
            if system_message:
                return self.handle_customize_gpt_response(gpt_response)
            self.log_gpt_response(identifier, gpt_response)
            chat_obj = self.add_chat(chat_from="BOT", chat_text=gpt_response, type="solution")
            if llm_automation:
                llm_automation.log_llm_response(chat_obj, rag_retrieval_result)
            return chat_obj
        return self.get_chatbot_disabled_response()

    def handle_customize_gpt_response(self, gpt_response):
        try:
            gpt_response_json = json.loads(gpt_response)
        except:
            return self.get_chatbot_disabled_response()
        chat_text = gpt_response_json.get("text", "")
        type = gpt_response_json.get("type", "valid_query" if chat_text else "gibberish").lower()
        question_query_user_chat_queryset = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, is_active=True, chat_from="USER"
        )
        # set user message type
        last_query_obj = question_query_user_chat_queryset.last()
        last_query_obj.type = type
        last_query_obj.save()
        if type == "valid_query":
            # send to admin if gpt could not figure out
            if not chat_text:
                reply_text = QUESTION_QUERY_PROMPT_DETAILS[self.query_obj.identifier]["admin_text"]
                return self.send_query_to_admin(reply_text)
            bot_response_type = "solution"
            return self.add_chat(chat_from="BOT", chat_text=chat_text, type=bot_response_type)
        self.add_in_spam(text=last_query_obj.chat_text, type=type)
        return self.handle_invalid_query()

    def handle_invalid_query(self):
        irrelevant_count = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id,
            is_active=True,
            chat_from="USER",
            type__in=["gibberish", "abuse", "irrelevant"],
        ).count()
        # close invalid query if more than once
        if irrelevant_count >= 2:
            reply_text = "I am sorry your query is not relevent to the question hence closing the chat, if you still have a doubt you can reach us at support@logiqids.com with a screenshot of the question and your query"
            chat_obj = self.add_chat(chat_from="BOT", chat_text=reply_text, type="email_us")
            self.resolve_query()
            return chat_obj
        chat_text = "We could not clearly understand your doubt"
        bot_response_type = "try_again"
        return self.add_chat(chat_from="BOT", chat_text=chat_text, type=bot_response_type)

    def add_in_spam(self, text, type):
        spam_chat_obj = QueryChatSpamResponses.objects.filter(text=text, type=type, is_active=True).last()
        if spam_chat_obj:
            spam_chat_obj.count += 1
            spam_chat_obj.save()
        else:
            QueryChatSpamResponses.objects.create(text=text, type=type)

    def check_in_spam(self, chat_obj):
        spam_chat_obj = QueryChatSpamResponses.objects.filter(
            text=chat_obj.chat_text, approved=True, is_active=True
        ).last()
        if spam_chat_obj:
            chat_obj.type = spam_chat_obj.type
            chat_obj.save()
            return True
        return False

    def set_user_message(self, text, image, date_obj):
        self.user_message = {"from": "USER", "text": text, "image": image, "profile_url": self.get_profile_url("USER")}
        self.user_message.update(self.get_date_time_formatted(date_obj=date_obj))

    def user_read_messages(self):
        QuestionQueriesChats.objects.filter(question_query_id=self.query_obj.id, is_active=True).exclude(
            chat_from="USER"
        ).update(is_read=True)

    def ask_user_to_elaboration(self):
        reply_text = QUESTION_QUERY_PROMPT_DETAILS[self.query_obj.identifier]["elaborate_text"]
        self.typing_enabled = True
        return self.add_chat(chat_from="BOT", chat_text=reply_text, type="explain")

    def user_chat_eloboratation_flow(self, identifier):
        question_query_chat_queryset = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, is_active=True
        )
        last_query_obj = question_query_chat_queryset.last()

        # ask for elaboration if chat doesn't exists
        if not last_query_obj:
            return self.ask_user_to_elaboration()

        # feedback selection response
        if last_query_obj.type == "feedback" and last_query_obj.chat_text.lower() == "no":
            # send to admin
            if question_query_chat_queryset.filter(type="valid_query").exists():
                reply_text = QUESTION_QUERY_PROMPT_DETAILS[self.query_obj.identifier]["admin_text"]
                return self.send_query_to_admin(reply_text)
            return self.ask_user_to_elaboration()

        # selection after irrelevant text
        if last_query_obj.type == "try_again_feedback":
            # if user wants to explain
            if last_query_obj.chat_text.lower() == "yes":
                reply_text = "Please make sure your query is related to the question or its solution"
                self.typing_enabled = True
                return self.add_chat(chat_from="BOT", chat_text=reply_text, type="explain")

        # one or two words response
        if len((last_query_obj.chat_text).split()) <= 2:
            prev_chat_list = question_query_chat_queryset.filter(chat_from="USER", type="explain").values_list(
                "chat_text", flat=True
            )
            few_words_chats = [x for x in prev_chat_list if len(x.split()) <= 2]
            if len(few_words_chats) == 1:
                reply_text = "Please describe your issue in detail for us to understand your doubt better"
                self.typing_enabled = True
                return self.add_chat(chat_from="BOT", chat_text=reply_text, type="explain")

        # check for spam in db
        if self.check_in_spam(last_query_obj):
            return self.handle_invalid_query()

        # classify query type using internal detectors instead of LLM
        context = self._build_query_context(last_query_obj)
        reclassification_result = self._classify_other_doubt_and_log_fallback(context, last_query_obj)
        if reclassification_result.should_reclassify:
            return self._route_reclassified_other_doubt(reclassification_result, last_query_obj)
        classification_service = QueryClassificationService.default()
        classification_result = classification_service.classify(context)
        return self._apply_classification_result(classification_result, last_query_obj)

    def _classify_other_doubt_and_log_fallback(self, context, last_query_obj):
        if self.query_obj.identifier != "other" or not OTHER_DOUBT_RECLASSIFICATION_ENABLED:
            return OtherDoubtReclassificationService.default(enabled=False).classify(context)
        service = OtherDoubtReclassificationService.default(enabled=OTHER_DOUBT_RECLASSIFICATION_ENABLED)
        result = service.classify(context)
        if not result.should_reclassify:
            self._log_other_doubt_reclassification(last_query_obj, result)
        return result

    def _route_reclassified_other_doubt(self, reclassification_result, last_query_obj):
        handler_identifier = reclassification_result.handler_identifier
        instructions = RECLASSIFIED_IDENTIFIER_TO_INSTRUCTIONS.get(handler_identifier, "")
        if handler_identifier in RECLASSIFIED_ADMIN_IDENTIFIERS:
            reply_text = QUESTION_QUERY_PROMPT_DETAILS[handler_identifier]["admin_text"]
            chat_obj = self.send_query_to_admin(reply_text)
            self._log_other_doubt_reclassification(last_query_obj, reclassification_result, handler_identifier)
            return chat_obj

        chat_obj = self.get_chat_gpt_response(
            identifier=handler_identifier,
            instructions=instructions,
            routed_identifier=handler_identifier,
        )
        self._log_other_doubt_reclassification(last_query_obj, reclassification_result, handler_identifier)
        return chat_obj

    def _log_other_doubt_reclassification(self, chat_obj, reclassification_result, final_handler_used=None):
        QueryBotLoggingService.log_other_doubt_reclassification(
            chat_obj, reclassification_result.to_log_data(final_handler_used=final_handler_used)
        )

    def _handle_low_relevance_with_legacy_flow(self):
        chat_obj = self.get_chat_gpt_response(system_message=CHATBOT_JSON_SYSTEM_MESSAGE)
        if not chat_obj:
            reply_text = QUESTION_QUERY_PROMPT_DETAILS[self.query_obj.identifier]["admin_text"]
            return self.send_query_to_admin(reply_text)
        return chat_obj

    def _build_query_context(self, last_query_obj):
        question_details = QuestionQueryService(self.query_obj.id).get_question_details()
        question_text = question_details["question_body"]["text"]
        solution_text = question_details["question_body"].get("solution") or ""
        options = [opt["text"] or "" for opt in question_details.get("options", [])]

        return QueryContext(
            user_text=last_query_obj.chat_text or "",
            question_text=question_text,
            solution_text=solution_text,
            options=options,
            identifier=self.query_obj.identifier,
            history_snippet=None,
            question_id=self.question_id,
        )

    def _apply_classification_result(self, classification_result, last_query_obj):
        self._record_classification_result(classification_result, last_query_obj)

        if classification_result.type != QueryType.VALID_QUERY:
            return self._handle_invalid_classified_query(last_query_obj)

        if self._is_low_relevance_valid_query(classification_result):
            return self._handle_low_relevance_with_legacy_flow()

        return self._answer_valid_classified_query(last_query_obj)

    def _record_classification_result(self, classification_result, last_query_obj):
        last_query_obj.type = classification_result.type.value
        last_query_obj.save()
        QueryBotLoggingService.log_classification(last_query_obj, classification_result)

    def _is_low_relevance_valid_query(self, classification_result):
        relevance_score = classification_result.scores.get("relevance_score")
        return relevance_score is not None and relevance_score < RAG_HIGH_RELEVANCE_THRESHOLD

    def _answer_valid_classified_query(self, last_query_obj):
        has_admin_responses = QueryGPTResponseBank.objects.filter(
            question_id=self.question_id, is_admin_response=True, is_active=True
        ).exists()
        embedding_svc = QuestionEmbeddingService.default()

        response_bank_chat = self._try_response_bank_solution(last_query_obj, has_admin_responses, embedding_svc)
        if response_bank_chat:
            return response_bank_chat

        if self.is_rag_enabled() and has_admin_responses:
            return self._answer_with_rag_or_llm(last_query_obj, embedding_svc)

        return self._answer_with_standard_llm_or_admin(instructions="")

    def _try_response_bank_solution(self, last_query_obj, has_admin_responses, embedding_svc):
        matching_rb, match_score = embedding_svc.find_best_matching_explain_intent_with_score(
            self.question_id, last_query_obj.chat_text, prefer_admin_responses=has_admin_responses
        )
        if not matching_rb:
            return None

        chat_obj = self.add_chat(chat_from="BOT", chat_text=matching_rb.response, type="solution")
        QueryBotLoggingService.log_response_bank_solution(
            chat_obj, match_score=match_score, question_id=self.question_id
        )
        return chat_obj

    def _answer_with_rag_or_llm(self, last_query_obj, embedding_svc):
        context = self._build_query_context(last_query_obj)
        rag_service = RAGService.default()
        answer, used_rag_only, retrieval_result = rag_service.get_answer_with_rag(context)

        if answer and used_rag_only:
            chat_obj = self.add_chat(chat_from="BOT", chat_text=answer, type="solution")
            QueryBotLoggingService.log_rag_solution(chat_obj, retrieval_result)
            return chat_obj

        rag_instructions = self._build_rag_instructions(retrieval_result)
        gpt_response = self.get_chat_gpt_response(instructions=rag_instructions, rag_retrieval_result=retrieval_result)
        if not gpt_response:
            reply_text = QUESTION_QUERY_PROMPT_DETAILS[self.query_obj.identifier]["admin_text"]
            return self.send_query_to_admin(reply_text)

        self._store_custom_explain_response(last_query_obj, gpt_response, embedding_svc)
        return gpt_response

    def _build_rag_instructions(self, retrieval_result):
        if not retrieval_result.documents:
            return ""
        return (
            CHATBOT_RAG_INSTRUCTIONS_PREFIX
            + "\n\n"
            + "\n\n---\n\n".join(doc.response_text for doc in retrieval_result.documents)
        )

    def _answer_with_standard_llm_or_admin(self, instructions, rag_retrieval_result=None):
        gpt_response = self.get_chat_gpt_response(instructions=instructions, rag_retrieval_result=rag_retrieval_result)
        if gpt_response:
            return gpt_response

        reply_text = QUESTION_QUERY_PROMPT_DETAILS[self.query_obj.identifier]["admin_text"]
        return self.send_query_to_admin(reply_text)

    def _store_custom_explain_response(self, last_query_obj, gpt_response, embedding_svc):
        response_text = getattr(gpt_response, "chat_text", "") if gpt_response else ""
        user_query_text = (last_query_obj.chat_text or "").strip()
        if not response_text or not user_query_text:
            return

        response_bank = QueryGPTResponseBank.objects.create(
            question_id=self.question_id,
            identifier="custom",
            response=response_text,
            weightage=0,
            query=user_query_text,
        )
        embedding_svc.create_embedding_for_explain_intent(self.question_id, response_bank)

    def _handle_invalid_classified_query(self, last_query_obj):
        self.add_in_spam(text=last_query_obj.chat_text, type=last_query_obj.type)
        return self.handle_invalid_query()

    def get_response_from_past_responses(self, identifier):
        if identifier in STORED_RESPONSE_IDENTIFIERS:
            solution_count = self.bot_solution_count()
            if solution_count < 1:
                response_bank_obj = (
                    QueryGPTResponseBank.objects.filter(
                        identifier=identifier, question_id=self.question_id, is_active=True, weightage__gt=0
                    )
                    .order_by("-is_admin_response", "-weightage")
                    .first()
                )
                if response_bank_obj:
                    return response_bank_obj.response
        return ""

    def log_gpt_response(self, identifier, gpt_response, weightage_diff=0, is_admin_response=False):
        if self.query_obj.chatbot and identifier in STORED_RESPONSE_IDENTIFIERS:
            response_obj, created_at = QueryGPTResponseBank.objects.get_or_create(
                question_id=self.question_id,
                response=gpt_response,
                identifier=identifier,
                is_active=True,
                is_admin_response=is_admin_response,
            )
            if weightage_diff:
                response_obj.weightage += weightage_diff
                response_obj.save()

    def on_feedback(self, feedback_text):
        identifier = self.get_identifier()
        last_query_obj = QuestionQueriesChats.objects.filter(
            question_query_id=self.query_obj.id, is_active=True, type="solution"
        ).last()
        if last_query_obj:
            weightage_diff = 0
            if feedback_text == "yes":
                weightage_diff += 1
            elif feedback_text == "no":
                weightage_diff -= 1
            if weightage_diff != 0:
                is_admin_response = last_query_obj.chat_from == "ADMIN"
                identifier = "custom"
                self.log_gpt_response(
                    identifier, last_query_obj.chat_text, weightage_diff, is_admin_response=is_admin_response
                )

    def is_image_present(self):
        question_details = self.get_question_details()
        question_body = question_details.get("question_body", {})
        if question_body.get("image") or question_body.get("group_image") or question_body.get("solution_image"):
            return True
        if any(option.get("image") for option in question_details.get("options", [])):
            return True
        return False
