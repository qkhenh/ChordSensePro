# 07 — AI Prompt System — Gọi AI để phân tích data

Dự án này dùng AI (OpenAI / Google AI Studio) để phân tích dữ liệu thương mại. Ví dụ: cho AI đọc danh sách mã HS code và hỏi "hàng hoá này thuộc ngành gì?"

Module `shared/infrastructure/service/` chứa toàn bộ hệ thống gọi AI.

---

## 1. Flow từ đầu đến cuối

```
PromptTemplate  →  BuiltPrompt  →  BaseRequestHandler  →  AI API  →  Structured Output
    (template)       (đã điền         (gọi API,              (JSON)      (Pydantic model)
                      data vào)        retry, batch)
```

---

## 2. `PromptTemplate` — Template câu hỏi

```python
class PromptTemplate(CustomBaseModel):
    system_prompt: str   # Vai trò của AI: "You are a trade analyst..."
    user_prompt: str     # Câu hỏi: "Classify this HS code: {hs_code}"

    def build(self, **kwargs) -> "BuiltPrompt":
        return BuiltPrompt(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt.format(**kwargs)
            # .format() điền giá trị vào {hs_code}, {buyer_name}, ...
        )
```

```python
class BuiltPrompt(CustomBaseModel):
    system_prompt: str
    user_prompt: str   # Đã điền đầy đủ, sẵn sàng gửi cho AI
```

---

## 3. `BaseRequestHandler` — Abstract class gọi AI

Mọi handler gọi AI đều kế thừa class này:

```python
class BaseRequestHandler(ABC, Generic[TInput, TOutput]):
    model: str                     # "gpt-4o" | "gemini-1.5-flash"
    output_schema: type[TOutput]   # Pydantic model cho output

    @abstractmethod
    def build_prompt(self, item: TInput) -> BuiltPrompt:
        # Subclass tạo prompt từ input
        pass

    def handle_one(self, item: TInput) -> TOutput:
        prompt = self.build_prompt(item)
        response = ai_client.complete(
            model=self.model,
            system=prompt.system_prompt,
            user=prompt.user_prompt,
            response_format=self.output_schema,  # Yêu cầu AI trả về JSON theo schema
        )
        return self.output_schema.model_validate_json(response)

    def handle_batch(self, items: list[TInput]) -> list[TOutput]:
        # Gọi nhiều items song song bằng ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.handle_one, item) for item in items]
            return [f.result() for f in futures]
```

---

## 4. Ví dụ: `HsCodeAnalystHandler`

```python
class HsCodeAnalystInput(BaseModel):
    hs_code: str
    description: str

class HsCodeAnalystOutput(BaseModel):
    industry: str
    sub_industry: str
    confidence: float

class HsCodeAnalystHandler(BaseRequestHandler[HsCodeAnalystInput, HsCodeAnalystOutput]):
    model = "gpt-4o"
    output_schema = HsCodeAnalystOutput

    TEMPLATE = PromptTemplate(
        system_prompt="You are an expert in international trade classification.",
        user_prompt=(
            "Given HS code {hs_code} with description '{description}', "
            "classify the industry and sub-industry."
        )
    )

    def build_prompt(self, item: HsCodeAnalystInput) -> BuiltPrompt:
        return self.TEMPLATE.build(
            hs_code=item.hs_code,
            description=item.description
        )
```

Cách dùng:
```python
handler = HsCodeAnalystHandler()

# Phân tích 1 item
result = handler.handle_one(HsCodeAnalystInput(hs_code="8471", description="Laptop"))
# HsCodeAnalystOutput(industry="Electronics", sub_industry="Computing", confidence=0.95)

# Phân tích nhiều items song song
results = handler.handle_batch([item1, item2, item3, ...])
```

---

## 5. Structured Output — AI trả về JSON chuẩn

Thay vì nhận text tự do từ AI rồi parse, ta dùng tính năng **structured output**:

```python
# Yêu cầu AI trả về JSON theo đúng schema Pydantic
response = openai_client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=HsCodeAnalystOutput,  # Pydantic model
)
result = response.choices[0].message.parsed  # Đã là HsCodeAnalystOutput object
```

AI sẽ **bị bắt buộc** trả về JSON đúng format. Nếu không đúng, API báo lỗi — không bao giờ nhận được text random.

---

## 6. ThreadPoolExecutor — Gọi song song

Gọi AI API mất thời gian (1-3 giây/request). Nếu có 100 items mà gọi tuần tự = 100-300 giây.

`handle_batch()` dùng `ThreadPoolExecutor` để gọi song song:

```python
with ThreadPoolExecutor(max_workers=5) as executor:
    # Submit 100 items cùng lúc
    futures = [executor.submit(self.handle_one, item) for item in items]
    # Đợi tất cả xong
    results = [f.result() for f in futures]
```

`max_workers=5` = chạy 5 requests cùng lúc. Không đặt quá cao → rate limit API.

---

## 7. Tóm lại

| Thành phần | Nhiệm vụ |
|-----------|---------|
| `PromptTemplate` | Giữ template câu hỏi, điền data vào |
| `BuiltPrompt` | Prompt đã sẵn sàng gửi |
| `BaseRequestHandler` | Gọi AI API, retry, batch processing |
| Subclass handler | Định nghĩa model, schema, cách build prompt |
| Structured output | AI trả về JSON theo Pydantic schema |
| `ThreadPoolExecutor` | Gọi nhiều items song song |
