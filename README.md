# ai-orchestrate

Небольшой локальный оркестратор Codex CLI. Jev выбирает начальный уровень, Codex CLI выполняет задачу на Luna/Sol, а указанные вами команды проверяют результат. Python-зависимостей нет.

## Быстрый старт в Windows PowerShell

1. Установите Python 3.10+, Git и Codex CLI. Войдите в Codex CLI обычным способом и проверьте `codex --version`, `git --version` и `python --version`.
2. Склонируйте репозиторий и перейдите в него:

   ```powershell
   git clone https://github.com/onix312/ai-orchestrate.git
   cd ai-orchestrate
   ```

   Репозиторий сейчас публичный. В нём не должно быть API-ключей, паролей или личных данных.
3. Проверьте окружение:

   ```powershell
   python -m ai_orchestrate doctor
   ```

   Команда показывает только наличие ключа, никогда его значение. Для live-маршрутизации нужен `TYPESAFE_API_KEY`.
4. Введите ключ скрытым вводом только в текущую сессию PowerShell:

   ```powershell
   $secureKey = Read-Host "TypeSafe API key" -AsSecureString
   $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
   try { $env:TYPESAFE_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer) }
   finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer); Remove-Variable secureKey, keyPointer }
   ```

   Не добавляйте ключ в код, README, задачи, логи или Git. Закройте окно PowerShell, чтобы убрать переменную из этой сессии.
5. Выберите отдельный Git-репозиторий для работы. Оркестратор требует хотя бы один коммит и чистое состояние Git, включая отсутствие неотслеживаемых файлов. Сначала можно проверить маршрут без сети и правок:

   ```powershell
   python -m ai_orchestrate run "Короткое описание задачи" --repo "C:/work/my-project" --check "python -m unittest discover -s tests" --dry-run --lane MEDIUM
   ```

6. Запустите задачу с одной или несколькими обязательными проверками:

   ```powershell
   python -m ai_orchestrate run "Добавь поиск по названию и тесты" --repo "C:/work/my-project" --check "python -m unittest discover -s tests" --check "ruff check ."
   ```

   Укажите только те команды, которые реально установлены и подходят проекту. Все проверки должны пройти, Codex должен завершиться успешно, а Jev должен подтвердить, что изменения соответствуют задаче. Иначе результат будет `INCOMPLETE`. По умолчанию максимум четыре прохода; допустимое значение `--max-attempts` — от 1 до 5.

## Поведение и границы

- Начальные полосы: SMALL → Luna low, MEDIUM → Luna medium, HIGH → Luna high, ESCALATE → Sol high. По умолчанию используются `gpt-6-luna` и `gpt-6-sol`; модель можно переопределить переменными `AI_ORCHESTRATE_LUNA_MODEL` и `AI_ORCHESTRATE_SOL_MODEL`.
- `--dry-run --lane ...` требует явно выбранную полосу и не вызывает Jev/Codex. `--lane` в обычном запуске запрещён: live-маршрут выбирает Jev.
- До запуска Codex проверяются все команды `--check` и чистота Git. После работы анализируется diff вместе со staged-изменениями и статусом, включая untracked файлы. Codex получает задачу через stdin; ключи TypeSafe/OpenRouter удаляются из его окружения.
- После сбоя повтор получает exit code и вывод проверок. Первый повтор поднимает уровень Luna, затем Jev выбирает только RETRY, ESCALATE, VERIFY или STOP. При таймауте новые попытки прекращаются. ОС может оставить дочерние процессы Codex работать; проверьте их вручную перед повторным запуском.
- Не передавайте в задаче секреты и не направляйте этот инструмент на репозитории с данными, которыми нельзя делиться с TypeSafe. Для решения Jev отправляются текст задачи, результаты проверок, ограниченный фрагмент diff и статус Git. Codex CLI получает только то, что нужно ему для работы.
- Программа не коммитит и не отправляет изменения. Просмотрите локальные изменения и решите сами, что делать дальше.

## Проверка проекта

```powershell
python -m unittest discover -s tests
```

Локально можно проверить запуск и dry-run. Живой запрос к TypeSafe и фактический запуск Luna/Sol требуют настроенных ключа/CLI и здесь не подтверждены.

Документация: [TypeSafe API](https://docs.typesafe.ai/api) · [Codex CLI reference](https://developers.openai.com/codex/cli/reference)
