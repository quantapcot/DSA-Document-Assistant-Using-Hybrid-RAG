"""40 Shared DSA Evaluation Questions (Q001-Q040) grounded verbatim in the 54 canonical source documents.

Distribution:
- Total: 40 shared questions (DRAFT status initially; identical across Profile A and Profile B)
- Topics: Array & Linked List (5), Stack & Queue (5), Tree, BST & AVL (6), Graph Algorithms (6),
          Searching & Sorting (6), Time & Space Complexity (4), Hashing (4), Algorithm Design Techniques (4)
- Languages: 24 Vietnamese (vi) + 16 English (en) (non-translated, independent source passages)
- Difficulties: 12 easy + 18 medium + 10 hard
- Question types: 4 each of all 10 required question types
"""
from __future__ import annotations

from typing import Any

# Each item declares exact (source_path, page, verbatim_snippet) evidence specifications
# resolved against live CanonicalDocument pages via locate_canonical_spans().
QUESTION_SPECS_40: list[dict[str, Any]] = [
    # =========================================================================
    # TOPIC 1: Array & Linked List (5: 3 VI, 2 EN)
    # =========================================================================
    {
        "query_id": "Q001",
        "query": "Mảng động (Dynamic Array) lưu trữ dữ liệu trong bộ nhớ như thế nào và cơ chế mở rộng dung lượng hoạt động ra sao?",
        "topic": "Array & Linked List",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "definition",
        "reference_answer": (
            "Mảng lưu trữ các phần tử tại các ô nhớ liền kề nhau (contiguous memory), cho phép truy xuất trực tiếp O(1) theo chỉ số. "
            "Mảng động (như std::vector trong C++ hoặc list trong Python) có khả năng tự động mở rộng (thường gấp đôi kích thước) khi đầy, "
            "nhờ đó thao tác thêm phần tử vào cuối đạt độ phức tạp trung bình (amortized) O(1)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Lưu trữ phần tử tại các ô nhớ liền kề và truy xuất O(1) theo chỉ số.", "evidence_mode": "alternative"},
            {"ku_id": "KU2", "description": "Tự động mở rộng (thường gấp đôi dung lượng) khi đầy để đạt amortized O(1) khi thêm vào cuối.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "alternative", "alt_group": "Q001_KU1",
                "source_path": "docx_Files/vie_docx/003_train_file_Vie.docx", "page": 2,
                "snippet": "Mảng lưu trữ dữ liệu tại các ô nhớ liền kề nhau (contiguous memory).\n\nTruy xuất phần tử (Access): O(1) nhờ vào chỉ số (index).",
            },
            {
                "ku_id": "KU1", "mode": "alternative", "alt_group": "Q001_KU1",
                "source_path": "pdf_Files/vie_pdf/020_train_file_Vie.pdf", "page": 3,
                "snippet": "Các phần tử trong vector được lưu trữ trong các ô nhớ liên tiếp\nvà truy cập được thông qua chỉ số.",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/003_train_file_Vie.docx", "page": 2,
                "snippet": "Mảng động (Vector trong C++, List trong Python) có khả năng tự động mở rộng (thường là gấp đôi kích thước) khi đầy, giúp thêm phần tử vào cuối (push_back/append) với độ phức tạp trung bình (amortized) là O(1).",
            },
        ],
    },
    {
        "query_id": "Q002",
        "query": "So sánh ưu điểm và nhược điểm giữa mảng và danh sách liên kết về tổ chức bộ nhớ, truy cập ngẫu nhiên và chèn xóa phần tử.",
        "topic": "Array & Linked List",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "comparison",
        "reference_answer": (
            "Mảng lưu trữ liền kề nên truy xuất ngẫu nhiên theo chỉ số mất O(1), nhưng thêm/xóa ở giữa mảng mất O(n) vì phải dịch chuyển các phần tử khác. "
            "Ngược lại, danh sách liên kết lưu trữ phân tán bằng con trỏ nên thêm/xóa ở đầu hoặc vị trí đã có con trỏ chỉ mất O(1) mà không cần dịch chuyển bộ nhớ, "
            "nhưng truy xuất ngẫu nhiên mất O(n) do phải duyệt tuần tự từ Head."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Mảng lưu liền kề: truy cập O(1) nhưng chèn/xóa giữa mảng mất O(n) do phải dịch chuyển.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Danh sách liên kết lưu phân tán qua con trỏ: chèn/xóa O(1) khi có con trỏ nhưng truy xuất ngẫu nhiên O(n).", "evidence_mode": "composite"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/003_train_file_Vie.docx", "page": 2,
                "snippet": "Truy xuất phần tử (Access): O(1) nhờ vào chỉ số (index).\n\nThêm/Xóa phần tử (Insert/Delete): O(n) ở giữa mảng vì phải dịch chuyển các phần tử khác.",
            },
            {
                "ku_id": "KU2", "mode": "composite", "alt_group": None,
                "source_path": "docx_Files/vie_docx/003_train_file_Vie.docx", "page": 3,
                "snippet": "Lưu trữ dữ liệu phân tán trong bộ nhớ (non-contiguous memory). Các phần tử (Node) liên kết với nhau bằng con trỏ.",
            },
            {
                "ku_id": "KU2", "mode": "composite", "alt_group": None,
                "source_path": "docx_Files/vie_docx/003_train_file_Vie.docx", "page": 3,
                "snippet": "Ưu điểm: Thêm/Xóa phần tử ở đầu hoặc giữa danh sách (nếu đã nắm giữ con trỏ tại vị trí đó) chỉ mất O(1), không cần dịch chuyển bộ nhớ.\n\nNhược điểm: Truy xuất ngẫu nhiên mất O(n) vì phải duyệt từ đầu (Head).",
            },
        ],
    },
    {
        "query_id": "Q003",
        "query": "Trong C++, cấu trúc một Node của danh sách liên kết đơn được khai báo như thế nào và cần lưu ý gì về quản lý bộ nhớ khi tạo node bằng toán tử new?",
        "topic": "Array & Linked List",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "code_interpretation",
        "reference_answer": (
            "Một struct Node cơ bản chứa trường dữ liệu (int data), con trỏ Node* next trỏ tới nút kế tiếp và hàm khởi tạo gán next = nullptr. "
            "Khi cấp phát nút động bằng new Node(...) trên Heap, lập trình viên cần tự giải phóng bộ nhớ bằng delete trong C++ thực tế để tránh rò rỉ bộ nhớ."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "struct Node gồm trường data, con trỏ Node* next và constructor khởi tạo next = nullptr.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Cần quản lý và giải phóng bộ nhớ (delete) khi cấp phát động bằng new trong C++.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/005_train_file_Vie.docx", "page": 3,
                "snippet": "struct Node {\n    int data;\n    Node* next;\n    Node(int val) { data = val; next = nullptr; }\n};",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/005_train_file_Vie.docx", "page": 3,
                "snippet": "Node* head = new Node(1);\nhead->next = new Node(2);\n// Chú ý: Cần quản lý bộ nhớ (delete) trong C++ thực tế.",
            },
        ],
    },
    {
        "query_id": "Q004",
        "query": "How does a Doubly Linked List differ structurally from a Singly Linked List, and what traversal capability does it enable?",
        "topic": "Array & Linked List",
        "language": "en",
        "difficulty": "medium",
        "question_type": "concept_explanation",
        "reference_answer": (
            "In a Singly Linked List, each node stores its data and a single pointer to the next node. "
            "A Doubly Linked List adds a second pointer to the previous node, which enables backward traversal through the list in addition to forward traversal."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Singly Linked List nodes store data and one pointer to the next node.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Doubly Linked List adds a pointer to the previous node, allowing backward traversal.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/end_docx/005_train_file_Eng.docx", "page": 3,
                "snippet": "Singly Linked List: Each node contains data and a pointer to the next node.",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/end_docx/005_train_file_Eng.docx", "page": 3,
                "snippet": "Doubly Linked List: Contains an additional pointer to the previous node, allowing backward traversal.",
            },
        ],
    },
    {
        "query_id": "Q005",
        "query": "Why does shrinking a Dynamic Array immediately when fill ratio drops to r = 1/2 hurt alternating insertions and deletions, and what threshold policy achieves amortized O(1) time with Theta(n) space?",
        "topic": "Array & Linked List",
        "language": "en",
        "difficulty": "hard",
        "question_type": "time_complexity",
        "reference_answer": (
            "If the array resizes directly back to fill ratio r = 1 when shrinking, alternating insertions and deletions can trigger expensive Theta(n) reallocations repeatedly. "
            "Instead, shrinking only when fill ratio falls below r_d to a larger target ratio r_i with r_d < r_i (for example r_d = 1/4, r_i = 1/2) ensures Theta(n) cheap operations occur before the next resize, "
            "maintaining Theta(n) space and amortized Theta(1) time per dynamic end operation."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Resizing to r = 1 on shrinkage causes poor performance under alternating insertions and deletions.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Shrinking when r < r_d to ratio r_i (with r_d < r_i, e.g., 1/4 to 1/2) guarantees Theta(n) cheap operations before the next resize.", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Maintains Theta(n) total space while achieving amortized O(1) append/pop at the end.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/002_train_file_Eng.pdf", "page": 4,
                "snippet": "- Attempt: if very empty, resize to r = 1. Alternating insertion and deletion could be bad...",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/002_train_file_Eng.pdf", "page": 4,
                "snippet": "- Idea! When r < rd, resize array to ratio ri where rd < ri (e.g., rd = 1/4, ri = 1/2)\n\n- Then Θ(n) cheap operations must be made before next expensive resize",
            },
            {
                "ku_id": "KU3", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/002_train_file_Eng.pdf", "page": 4,
                "snippet": "- However, can be very wasteful in space. Want size of data structure to stay Θ(n)",
            },
        ],
    },

    # =========================================================================
    # TOPIC 2: Stack & Queue (5: 3 VI, 2 EN)
    # =========================================================================
    {
        "query_id": "Q006",
        "query": "Hàng đợi hai đầu (Deque) khác gì hàng đợi thông thường và hỗ trợ những thao tác cơ sở nào?",
        "topic": "Stack & Queue",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "definition",
        "reference_answer": (
            "Hàng đợi hai đầu (DeQueue) là cấu trúc dữ liệu dạng hàng đợi nhưng hỗ trợ phép bổ sung và loại bỏ phần tử ở cả đầu lẫn cuối. "
            "Các hàm cơ sở gồm insertFirst(o), insertLast(o), removeFirst() và removeLast() (cùng các hàm phụ trợ first(), last(), size(), isEmpty(), create())."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Deque hỗ trợ thêm và loại bỏ phần tử ở cả hai đầu (đầu và cuối).", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Bốn thao tác cơ sở: insertFirst(o), insertLast(o), removeFirst(), removeLast().", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/008_train_file_Vie.pdf", "page": 17,
                "snippet": "- Hàng đợi hai đầu là một cấu trúc dữliệu dạng hàng đợi\n\nnhưng nó hỗtrợphép bổsung và loại bỏởcảđầu và cuối",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/008_train_file_Vie.pdf", "page": 17,
                "snippet": "- insertFirst(o)\n\n                - insertLast(o) :\n\n                - removeFirst()\n\n                - removeLast()",
            },
        ],
    },
    {
        "query_id": "Q007",
        "query": "Trong mã giả cài đặt Queue bằng danh sách móc nối với hai con trỏ F (front) và R (rear), thao tác ENQUEUE và DEQUEUE xử lý trường hợp danh sách rỗng hoặc chỉ có một phần tử như thế nào?",
        "topic": "Stack & Queue",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "pseudocode_interpretation",
        "reference_answer": (
            "Trong ENQUEUE(F, R, X), sau khi tạo nút mới p với INFO(p):=X và LINK(p):=Null, nếu hàng đợi đang rỗng (F = Null) thì gán cả F := R := p; ngược lại nối LINK(R) := p rồi cập nhật R := p. "
            "Trong DEQUEUE(F, R, Y), lấy nút đầu p := F và Y := INFO(p); nếu hàng đợi chỉ có một phần tử ((F = R) and (F <> Null)) thì đặt cả F := R := Null, ngược lại dịch F := LINK(p) và giải phóng nút p."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "ENQUEUE khi rỗng (F = Null) gán F := R := p; khi không rỗng nối LINK(R) := p và R := p.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "DEQUEUE khi chỉ có 1 phần tử (F = R và F <> Null) đặt F := R := Null; ngược lại F := LINK(p) và Dispose(p).", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/008_train_file_Vie.pdf", "page": 16,
                "snippet": "2. {Danh sách đã cho rỗng} if F = Null then F:= R:= p;\n\n                3. else LINK(R) := p; R:= p;",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/008_train_file_Vie.pdf", "page": 16,
                "snippet": "if (F = R) and (F <> Null) then F:= R:= Null;\n\n                2. else F:= LINK(p);\n\n                3. Call Dispose(p) ;",
            },
        ],
    },
    {
        "query_id": "Q008",
        "query": "Tại sao khi cài đặt Queue (FIFO) trong Python nên dùng collections.deque thay vì list, và trong C++ cần cấu hình std::priority_queue như thế nào để tạo Min-Heap?",
        "topic": "Stack & Queue",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "code_interpretation",
        "reference_answer": (
            "Trong Python, Queue phải dùng collections.deque (với append và popleft) để thao tác xóa phần tử ở đầu đạt O(1). "
            "Trong C++, std::priority_queue mặc định là Max-Heap; để tạo Min-Heap cần khai báo tường minh std::priority_queue<int, std::vector<int>, std::greater<int>>, "
            "giúp truy xuất phần tử nhỏ nhất qua top() trong O(1) và xóa bằng pop() trong O(log n)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Trong Python dùng collections.deque và popleft() để xóa ở đầu hàng đợi trong O(1).", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "C++ std::priority_queue mặc định là Max-Heap; dùng std::greater<int> với vector<int> để tạo Min-Heap.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/005_train_file_Vie.docx", "page": 4,
                "snippet": "# Queue (FIFO - Phải dùng deque để đạt O(1) khi xóa ở đầu)\nq = deque()\nq.append(1)\nq.popleft()",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/007_train_file_Vie.docx", "page": 4,
                "snippet": "// C++ mặc định là Max-Heap. Để dùng Min-Heap cần cấu hình:\nstd::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;",
            },
        ],
    },
    {
        "query_id": "Q009",
        "query": "Compare implementing a Priority Queue with an unordered dynamic array versus a sorted dynamic array in terms of insert and delete_max complexities, and identify which sorting algorithm each corresponds to.",
        "topic": "Stack & Queue",
        "language": "en",
        "difficulty": "medium",
        "question_type": "comparison",
        "reference_answer": (
            "An unordered dynamic array supports insert(x) in amortized O(1) time by appending to the end, but delete_max() takes O(n) to scan for the maximum — corresponding to Selection Sort. "
            "A sorted dynamic array supports delete_max() in amortized O(1) from the end, but insert(x) takes O(n) to shift into sorted position — corresponding to Insertion Sort."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Unordered dynamic array PQ: amortized O(1) insert, O(n) delete_max; yields Selection Sort.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Sorted dynamic array PQ: O(n) insert, amortized O(1) delete_max; yields Insertion Sort.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/008_train_file_Eng.pdf", "page": 2,
                "snippet": "insert(x): append x to end in amortized O(1) time\n\n   - delete max(): ﬁnd max in O(n), swap max to the end and remove\n\n- insert is quick, but delete max is slow\n\n- Priority queue sort is selection sort!",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/008_train_file_Eng.pdf", "page": 2,
                "snippet": "insert(x): append x to end, swap down to sorted position in O(n) time\n\n   - delete max(): delete from end in O(1) amortized\n\n- delete max is quick, but insert is slow\n\n- Priority queue sort is insertion sort!",
            },
        ],
    },
    {
        "query_id": "Q010",
        "query": "What are the governing ordering principles and classic practical applications of Stacks versus Queues?",
        "topic": "Stack & Queue",
        "language": "en",
        "difficulty": "easy",
        "question_type": "application_scenario",
        "reference_answer": (
            "A Stack follows the LIFO (Last In First Out) principle and is used in Undo/Redo features, browser history, and valid parentheses checking. "
            "A Queue follows the FIFO (First In First Out) principle and is used in task scheduling and Breadth-First Search (BFS)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Stack uses LIFO; applied in Undo/Redo, browser history, and valid parentheses checking.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Queue uses FIFO; applied in task scheduling and BFS.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/end_docx/005_train_file_Eng.docx", "page": 4,
                "snippet": "Stack: LIFO (Last In First Out) principle. Applications: Undo/Redo features, Browser history, Valid Parentheses checking.",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/end_docx/005_train_file_Eng.docx", "page": 4,
                "snippet": "Queue: FIFO (First In First Out) principle. Applications: Task scheduling, BFS (Breadth-First Search) algorithms.",
            },
        ],
    },

    # =========================================================================
    # TOPIC 3: Tree, BST & AVL (6: 4 VI, 2 EN)
    # =========================================================================
    {
        "query_id": "Q011",
        "query": "Trình bày thứ tự thăm nút của ba phép duyệt cây nhị phân (Duyệt trước, Duyệt giữa, Duyệt sau) và phép duyệt nào cho ra dãy khóa tăng dần trên BST?",
        "topic": "Tree, BST & AVL",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "definition",
        "reference_answer": (
            "Ba phép duyệt cây nhị phân gồm: Duyệt trước (Pre-order: Gốc - Trái - Phải), Duyệt giữa (In-order: Trái - Gốc - Phải) và Duyệt sau (Post-order: Trái - Phải - Gốc). "
            "Trên cây tìm kiếm nhị phân (BST), phép duyệt giữa (In-order / LNR) sẽ cho ra mảng các khóa theo thứ tự tăng dần."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Ba thứ tự duyệt: Pre-order (Gốc-Trái-Phải), In-order (Trái-Gốc-Phải), Post-order (Trái-Phải-Gốc).", "evidence_mode": "alternative"},
            {"ku_id": "KU2", "description": "Phép duyệt In-order trên BST cho ra dãy giá trị tăng dần.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "alternative", "alt_group": "Q011_KU1",
                "source_path": "docx_Files/vie_docx/004_train_file_Vie.docx", "page": 3,
                "snippet": "Duyệt cây (Traversal): In-order (Trái-Gốc-Phải - cho ra mảng tăng dần), Pre-order (Gốc-Trái-Phải), Post-order (Trái-Phải-Gốc).",
            },
            {
                "ku_id": "KU1", "mode": "alternative", "alt_group": "Q011_KU1",
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 70,
                "snippet": "o Duyệt trước: Gốc - Trái - Phải (Node - Left - Right)\n         o Duyệt giữa: Trái - Gốc - Phải (Left - Node - Right)\n         o Duyệt sau: Trái - Phải - Gốc (Left - Right - Node)",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/004_train_file_Vie.docx", "page": 3,
                "snippet": "In-order (Trái-Gốc-Phải - cho ra mảng tăng dần)",
            },
        ],
    },
    {
        "query_id": "Q012",
        "query": "Trên cây tìm kiếm 2-3, giải thuật SEARCH-2-3(T, X) rẽ nhánh qua các con trỏ LPTR, MPTR, RPTR như thế nào và các trường hợp ban đầu khi bổ sung nút mới là gì?",
        "topic": "Tree, BST & AVL",
        "language": "vi",
        "difficulty": "hard",
        "question_type": "multi_evidence_reasoning",
        "reference_answer": (
            "Khi duyệt các nút trong (TYPE(p) = 1), SEARCH-2-3(T, X) đi sang LPTR(p) nếu X <= LDATA(p), sang MPTR(p) nếu X <= MDATA(p), ngược lại sang RPTR(p) nếu RPTR(p) khác NULL; "
            "khi xuống tới nút lá thì kiểm tra VALUE(p) = X để trả về p hoặc NULL. "
            "Khi bổ sung vào cây 2-3 cần xét các trường hợp: cây ban đầu rỗng, cây chỉ có 1 nút (tạo thêm nút nhánh), hoặc bổ sung vào nút đang có 2 con bằng cách so sánh giá trị để đặt đúng vị trí (và tách nút khi nút đã có 3 con)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Rẽ nhánh tại nút trong: X <= LDATA -> LPTR, X <= MDATA -> MPTR, ngược lại RPTR nếu khác NULL.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Khi đến nút lá so sánh VALUE(p) = X để trả về p hoặc NULL.", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Phép bổ sung trên cây 2-3 xét các trường hợp cây rỗng, cây 1 nút, và nút cha đang có 2 con.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/014_train_file_Vie.pdf", "page": 3,
                "snippet": "if X <= LDATA(p) then p:= LPTR(p);\n\n                else if X <= MDATA(p) then p:= MPTR(p);\n\n                else if RPTR(p) < > NULL then p:= RPTR(p)",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/014_train_file_Vie.pdf", "page": 3,
                "snippet": "{Tìm được đến 1 nút lá}\n\n               if VALUE(p) = X then SEARCH-2-3 := p; else SEARCH-2-3 := NULL;",
            },
            {
                "ku_id": "KU3", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/014_train_file_Vie.pdf", "page": 4,
                "snippet": "- Cây ban đầu rỗng\n\n- Cây ban đầu chỉcó 1 nút: Sau khi bổsung, cây có thêm\n\nmột nút nhánh",
            },
        ],
    },
    {
        "query_id": "Q013",
        "query": "Tại sao Heap thường được cài đặt bằng mảng thay vì con trỏ và quy tắc biểu diễn cây nhị phân hoàn chỉnh lên mảng thực hiện như thế nào?",
        "topic": "Tree, BST & AVL",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "application_scenario",
        "reference_answer": (
            "Heap là một cây nhị phân hoàn chỉnh nên thường được cài đặt bằng mảng để tiết kiệm bộ nhớ con trỏ. "
            "Các nút trên cây được đánh số lần lượt từ mức 1, hết mức này sang mức khác, từ trái sang phải, và lưu thông tin nút thứ i vào phần tử V[i] của vector lưu trữ."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Heap là cây nhị phân hoàn chỉnh, cài bằng mảng giúp tiết kiệm bộ nhớ con trỏ.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Đánh số nút theo từng mức từ trên xuống, trái sang phải và lưu vào V[i].", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/004_train_file_Vie.docx", "page": 4,
                "snippet": "Heap là một Cây nhị phân hoàn chỉnh (Complete Binary Tree), thường được cài đặt bằng Mảng (Array) để tiết kiệm bộ nhớ con trỏ.",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/007_train_file_Vie.pdf", "page": 22,
                "snippet": "z Đánh sốcác nút trên cây theo trình tựtừmức 1, hết mức\n\nnày đến mức khác, từtrái sang phải\n\nz Lưu trữtrong vector lưu trữV theo nguyên tắc phần tửV[i]\n\nsẽlưu thông tin của nút được đánh sối",
            },
        ],
    },
    {
        "query_id": "Q014",
        "query": "Trong hàm insert_case5 của cây Đỏ-Đen (Red-Black Tree), khi nút cha P là đỏ, nút bác U là đen và nút mới N nằm cùng phía với P so với nút ông G, hàm thực hiện đổi màu và phép quay nào?",
        "topic": "Tree, BST & AVL",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "code_interpretation",
        "reference_answer": (
            "Hàm insert_case5 đổi màu nút cha P thành BLACK và nút ông G (grandparent) thành RED. "
            "Sau đó thực hiện phép quay đơn tại nút ông G: nếu N là con trái của P và P là con trái của G thì gọi rotate_right(grandparent(n)), "
            "ngược lại (cùng phía phải) thì gọi rotate_left(grandparent(n)) để khôi phục tính chất cây Đỏ-Đen."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Đổi màu nút cha P thành BLACK và nút ông G thành RED.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Thực hiện rotate_right(grandparent(n)) nếu cùng phía trái, hoặc rotate_left(grandparent(n)) nếu cùng phía phải.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/004_train_file_Vie.pdf", "page": 121,
                "snippet": "void insert_case5(struct node *n) {\n            n->parent->color = BLACK;\n            grandparent(n)->color = RED;",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/004_train_file_Vie.pdf", "page": 121,
                "snippet": "if (n == n->parent->left && n->parent == grandparent(n)->left) {\n            rotate_right(grandparent(n));\n            } else {\n            /* Here, n == n->parent->right && n->parent == grandparent(n)->right */\n            rotate_left(grandparent(n));",
            },
        ],
    },
    {
        "query_id": "Q015",
        "query": "In a binary tree, how do you find the traversal-order successor of a node X, and how is a non-leaf node X deleted without violating traversal order?",
        "topic": "Tree, BST & AVL",
        "language": "en",
        "difficulty": "medium",
        "question_type": "concept_explanation",
        "reference_answer": (
            "To find the successor of node X: if X has a right child, return the first (leftmost) node in X's right subtree; "
            "otherwise, walk upward and return the lowest ancestor of X for which X lies in its left subtree (O(h) time). "
            "To delete a non-leaf node X: swap its item with its predecessor (if X has a left child) or with its successor (if X has a right child) and recursively delete down to a leaf in O(h) time."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Successor: first node of right subtree if right child exists, else lowest ancestor having X in its left subtree.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Deleting a non-leaf node swaps its item with predecessor/successor and recurses until reaching a leaf in O(h) time.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/010_train_file_Eng.pdf", "page": 2,
                "snippet": "- Find successor of node <X> in the traversal order (predecessor is symmetric)\n\n- If <X> has right child, return ﬁrst of right subtree\n\n- Otherwise, return lowest ancestor of <X> for which <X> is in its left subtree",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/010_train_file_Eng.pdf", "page": 3,
                "snippet": "∗ If <X> has a left child, swap items with the predecessor of <X> and recurse\n∗ Otherwise <X> has a right child, swap items with the successor of <X> and recurse",
            },
        ],
    },
    {
        "query_id": "Q016",
        "query": "In an AVL tree, how is node skew defined, why does height-balance guarantee O(log n) height, and what rotations rebalance a node B with skew +2 depending on the skew of its right child F?",
        "topic": "Tree, BST & AVL",
        "language": "en",
        "difficulty": "hard",
        "question_type": "multi_evidence_reasoning",
        "reference_answer": (
            "Skew of a node is height(right subtree) minus height(left subtree); a node is height-balanced if its skew is -1, 0, or 1. "
            "The minimum nodes F(h) in a height-h AVL tree satisfies F(h) = 1 + F(h-1) + F(h-2) >= 2*F(h-2) >= 2^(h/2), guaranteeing h = O(log n). "
            "When node B has skew 2 (right child F exists): if skew(F) is 0 or 1, perform a single left rotation on B; if skew(F) is -1, perform a right rotation on F followed by a left rotation on B."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Skew = height(right) - height(left); height-balanced requires skew in {-1, 0, 1}.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Recurrence F(h) = 1 + F(h-1) + F(h-2) >= 2^(h/2) proves height h = O(log n).", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "For skew(B)=2: left rotate B if skew(F) in {0, 1}; right rotate F then left rotate B if skew(F) = -1.", "evidence_mode": "composite"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/009_train_file_Eng.pdf", "page": 2,
                "snippet": "- Let skew of a node be the height of its right subtree minus that of its left subtree\n\n- Then a node is height-balanced if its skew is −1, 0, or 1",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/009_train_file_Eng.pdf", "page": 2,
                "snippet": "- Claim: A binary tree with height-balanced nodes has height h = O(log n) (i.e., n = 2Ω(h))\n\n- Proof: Sufﬁces to show fewest nodes F (h) in any height h tree is F (h) = 2Ω(h)\n\nF (h) ≥ 2h/2\nF (0) = 1, F (1) = 2, F (h) = 1+F (h−1)+F (h−2) ≥ 2F (h−2)",
            },
            {
                "ku_id": "KU3", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/009_train_file_Eng.pdf", "page": 3,
                "snippet": "- Case 1: skew of <F> is 0 or Case 2: skew of <F> is 1\n\n∗ Perform a left rotation on <B>",
            },
            {
                "ku_id": "KU3", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/009_train_file_Eng.pdf", "page": 3,
                "snippet": "- Case 3: skew of <F> is −1, so the left child <D> of <F> exists\n\n∗ Perform a right rotation on <F>, then a left rotation on <B>",
            },
        ],
    },

    # =========================================================================
    # TOPIC 4: Graph Algorithms (6: 4 VI, 2 EN)
    # =========================================================================
    {
        "query_id": "Q017",
        "query": "So sánh tính chất của ma trận kề trên đồ thị vô hướng với đồ thị có hướng, và nêu ưu điểm của biểu diễn đồ thị bằng danh sách kề.",
        "topic": "Graph Algorithms",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "comparison",
        "reference_answer": (
            "Với đồ thị vô hướng, ma trận kề đối xứng, tổng các phần tử bằng 2 lần số cạnh và tổng hàng/cột k là bậc của đỉnh k. "
            "Với đồ thị có hướng, ma trận kề có thể không đối xứng, tổng các phần tử bằng số cạnh, tổng hàng k là bậc ra và tổng cột k là bậc vào của đỉnh k. "
            "Biểu diễn bằng danh sách kề tiết kiệm không gian lưu trữ (chỉ lưu các cạnh và đỉnh kề) và truy cập nhanh các đỉnh kề, rất thuận lợi cho BFS và DFS."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Ma trận kề đồ thị vô hướng đối xứng, tổng = 2*số cạnh; đồ thị có hướng có thể không đối xứng, hàng là bậc ra, cột là bậc vào.", "evidence_mode": "composite"},
            {"ku_id": "KU2", "description": "Danh sách kề tiết kiệm bộ nhớ và truy cập nhanh các đỉnh kề, tối ưu cho BFS/DFS.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 105,
                "snippet": "- Ma trận kề là ma trận đối xứng\n- Tổng các phần tử trên ma trận bằng 2 lần số cạnh\n- Tổng các phần tử trên hàng thứ k hoặc cột thứ k là bậc của đỉnh k",
            },
            {
                "ku_id": "KU1", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 105,
                "snippet": "- Ma trận kề có thể không đối xứng\n- Tổng các phần tử trên ma trận bằng số cạnh\n- Tổng các phần tử trên hàng thứ k là tổng số bậc ra của đỉnh k\n- Tổng các phần tử trên cột thứ k là tổng số bậc vào của đỉnh k",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 107,
                "snippet": "- Tiết kiệm không gian lưu trữ: chỉ lưu các cạnh và đỉnh kề của mỗi đỉnh\n- Truy cập nhanh đến đỉnh kề: dễ dàng truy cập đến các đỉnh kề của một đỉnh cụ\nthể, hữu ích trong việc thực hiện các thuật toán đồ thị như duyệt theo chiều\nrộng (BFS) hoặc duyệt theo chiều sâu (DFS).",
            },
        ],
    },
    {
        "query_id": "Q018",
        "query": "Trình bày thuật toán DFS tìm tất cả các cạnh cầu (bridge) của đồ thị vô hướng offline bằng hai mảng tin[] và fup[], kèm điều kiện nhận biết cầu và độ phức tạp.",
        "topic": "Graph Algorithms",
        "language": "vi",
        "difficulty": "hard",
        "question_type": "algorithm_steps",
        "reference_answer": (
            "Cầu là cạnh mà nếu bỏ đi thì số thành phần liên thông của đồ thị tăng lên. "
            "Thuật toán DFS offline chạy trong O(n + m): gán tin[v] là thời điểm vào đỉnh v và xây dựng mảng fup[v] = min(tin[v], tin[p] của các cạnh ngược, fup[to] của các đỉnh con to). "
            "Cạnh (v, to) trong cây DFS là một cầu khi và chỉ khi fup[to] > tin[v] (từ to và cây con của to không có đường quay ngược tới v hoặc tổ tiên của v)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Định nghĩa cầu (bỏ cạnh làm tăng số thành phần liên thông) và độ phức tạp DFS offline O(n+m).", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Sử dụng thời điểm thăm tin[v] và mảng fup[v] = min(tin[v], tin[p], fup[to]).", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Điều kiện cạnh (v, to) là cầu khi fup[to] > tin[v].", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/002_train_file_Vie.pdf", "page": 3,
                "snippet": "Cầu của đồ thị G là cạnh mà nếu bỏ nó số thành phần của G sẽ tăng.",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/002_train_file_Vie.pdf", "page": 3,
                "snippet": "Gọi tin[v] - thời điểm vào đỉnh v khi duyệt theo chiều sâu.\n\nXây dựng mảng bổ trợ fup[v]:",
            },
            {
                "ku_id": "KU3", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/002_train_file_Vie.pdf", "page": 4,
                "snippet": "if (fup[to] > tin[v])\n\n                IS_BRIDGE(v,to);",
            },
        ],
    },
    {
        "query_id": "Q019",
        "query": "Trong giải thuật mã giả DFS(G, v), hai loại cung DISCOVERY và BACK được gán nhãn dựa trên điều kiện nào khi duyệt các cung liên thuộc với đỉnh v?",
        "topic": "Graph Algorithms",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "pseudocode_interpretation",
        "reference_answer": (
            "Sau khi đánh dấu đỉnh v là VISITED, giải thuật duyệt mọi cung e liên thuộc với v đang ở trạng thái UNEXPLORED và lấy đỉnh đối w = opposite(v, e). "
            "Nếu đỉnh w chưa được thăm (getLabel(w) = UNEXPLORED) thì gán nhãn cung e là DISCOVERY (cung khám phá) và gọi đệ quy DFS(G, w); "
            "ngược lại (w đã được thăm) thì gán nhãn cung e là BACK (cung quay lui)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Đánh dấu v là VISITED và xét các cung e có nhãn UNEXPLORED với đỉnh đối w = opposite(v, e).", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Nếu w chưa thăm thì đánh dấu e là DISCOVERY và đệ quy DFS(G, w); nếu w đã thăm thì đánh dấu e là BACK.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/016_train_file_Vie.pdf", "page": 9,
                "snippet": "setLabel(v, VISITED) // đỉnh v đã được thăm\n\n                for all e ∈G.incidentEdges(v)\n\n                if getLabel(e) = UNEXPLORED\n\n                w ←opposite(v,e)",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/016_train_file_Vie.pdf", "page": 9,
                "snippet": "if getLabel(w) = UNEXPLORED\n\n                setLabel(e, DISCOVERY)\n                DFS(G, w)\n                else\n\n                setLabel(e, BACK)",
            },
        ],
    },
    {
        "query_id": "Q020",
        "query": "Trong đoạn code C++ cài đặt thuật toán Dijkstra sử dụng priority_queue<pair<int,int>>, tại sao lại lấy cur_d = -q.top().first và kiểm tra điều kiện if (cur_d > d[v]) continue?",
        "topic": "Graph Algorithms",
        "language": "vi",
        "difficulty": "hard",
        "question_type": "code_interpretation",
        "reference_answer": (
            "Do std::priority_queue trong C++ mặc định là Max-Heap, chương trình lưu khoảng cách âm vào phần tử first của pair nên khi lấy ra cần đảo dấu cur_d = -q.top().first để mô phỏng Min-Heap. "
            "Vì mỗi lần tối ưu d[to] đều đẩy một bản sao mới vào hàng đợi ưu tiên mà không xóa bản cũ, điều kiện if (cur_d > d[v]) continue giúp bỏ qua các trạng thái cũ không còn tối ưu trước khi duyệt các cạnh kề của v."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "priority_queue mặc định là Max-Heap nên đảo dấu (-q.top().first) để lấy đỉnh có khoảng cách nhỏ nhất.", "evidence_mode": "composite"},
            {"ku_id": "KU2", "description": "Câu lệnh if (cur_d > d[v]) continue bỏ qua các phần tử cũ có khoảng cách lớn hơn giá trị tối ưu hiện tại d[v].", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/004_train_file_Vie.pdf", "page": 188,
                "snippet": "priority_queue < pair<int,int> > q;\n      q.push (make_pair (0, s));\n      while (!q.empty()) {\n            int v = q.top().second, cur_d = -q.top().first;",
            },
            {
                "ku_id": "KU1", "mode": "composite", "alt_group": None,
                "source_path": "docx_Files/vie_docx/007_train_file_Vie.docx", "page": 4,
                "snippet": "// C++ mặc định là Max-Heap.",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/004_train_file_Vie.pdf", "page": 188,
                "snippet": "q.pop();\n            if (cur_d > d[v]) continue;",
            },
        ],
    },
    {
        "query_id": "Q021",
        "query": "Why does an adjacency-list graph representation require Theta(|V| + |E|) space, and why does storing a Shortest Paths Tree via parent pointers take O(|V|) instead of Omega(|V|^2) space?",
        "topic": "Graph Algorithms",
        "language": "en",
        "difficulty": "medium",
        "question_type": "space_complexity",
        "reference_answer": (
            "An adjacency list stores a set Adj of size Theta(|V|) mapping each vertex u to its neighbor list Adj(u) of size Theta(deg(u)); "
            "by the handshaking lemma, the sum of deg(u) over all vertices is at most 2|E|, giving Theta(|V| + |E|) total space. "
            "Returning every shortest path explicitly could take Omega(|V|^2) time/space because paths can have length Omega(|V|), "
            "whereas storing only one parent pointer P(v) per vertex encodes the entire Shortest Paths Tree in O(|V|) space."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Adjacency list uses Theta(|V|) for Adj plus sum of deg(u) <= 2|E| by handshaking lemma, totaling Theta(|V| + |E|) space.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Storing parent pointers P(v) encodes all shortest paths from s in O(|V|) size instead of Omega(|V|^2).", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/007_train_file_Eng.pdf", "page": 2,
                "snippet": "For the common representations, Adj has size Θ(|V |), while each Adj(u) has size Θ(deg(u))\nP\n- Since\n            u∈V deg(u) ≤ 2|E| by handshaking lemma, graph storable in Θ(|V | + |E|) space",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/007_train_file_Eng.pdf", "page": 3,
                "snippet": "Many paths could have length Ω(|V |), so returning every path could require Ω(|V |2) time\n\n- Instead, for all v ∈ V , store its parent P (v): second to last vertex on a shortest path from s\n\n- Let P (s) be null (no second to last vertex on shortest path from s to s)\n\n   - Set of parents comprise a shortest paths tree with O(|V |) size!",
            },
        ],
    },
    {
        "query_id": "Q022",
        "query": "How does Bellman-Ford use graph duplication into |V| + 1 levels and negative-cycle witnesses to compute shortest paths and detect vertices with distance -infinity?",
        "topic": "Graph Algorithms",
        "language": "en",
        "difficulty": "hard",
        "question_type": "algorithm_steps",
        "reference_answer": (
            "Bellman-Ford constructs a layered DAG G' with |V| + 1 levels where vertex v_k represents reaching v from s using at most k edges, adding zero-weight edges (v_{k-1}, v_k) and weight-w(u,v) edges (u_{k-1}, v_k), then runs DAG Relaxation from s_0. "
            "Each vertex gets d(s, v) = delta(s_0, v_{|V|-1}). Any vertex u with delta(s_0, u_{|V|}) < delta(s_0, u_{|V|-1}) is a negative-cycle witness; "
            "for every witness u and every vertex v reachable from u in G, Bellman-Ford sets d(s, v) = -infinity."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Graph duplication creates |V|+1 levels forming a DAG G' where level k represents paths of <= k edges, solved by DAG Relaxation.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "A negative-cycle witness u satisfies delta_{|V|}(s, u) < delta_{|V|-1}(s, u).", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Every vertex v reachable from any witness u in G has shortest-path distance set to -infinity.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/019_train_file_Eng.pdf", "page": 3,
                "snippet": "- |V | + 1 levels: vertex vk in level k represents reaching vertex v from s using ≤ k edges\n\n- If edges only increase in level, resulting graph is a DAG!",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/019_train_file_Eng.pdf", "page": 2,
                "snippet": "- If δ|V |(s, v) < δ|V |−1(s, v)\n\n∗ there exists a shorter non-simple path to v, so δ|V |(s, v) = −∞\n∗ call v a (negative cycle) witness",
            },
            {
                "ku_id": "KU3", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/019_train_file_Eng.pdf", "page": 3,
                "snippet": "- For each witness u ∈ V where δ(s0, u|V |) < δ(s0, u|V |−1):\n\n- For each vertex v reachable from u in G:\n\n           ∗ set d(s, v) = −∞",
            },
        ],
    },

    # =========================================================================
    # TOPIC 5: Searching & Sorting (6: 4 VI, 2 EN)
    # =========================================================================
    {
        "query_id": "Q023",
        "query": "Trình bày các bước hoạt động của thuật toán Sắp xếp chèn (Insertion Sort) theo cách ví von xếp các lá bài.",
        "topic": "Searching & Sorting",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "algorithm_steps",
        "reference_answer": (
            "Insertion Sort hoạt động giống xếp bài trên tay: bắt đầu với tay trái trống, lần lượt lấy từng phần tử (lá bài) chưa sắp xếp và chèn nó vào vị trí đúng trong phần đã sắp xếp bên trái "
            "bằng cách so sánh ngược từ phải sang trái. Lặp lại cho đến khi mọi phần tử đều được chèn vào đúng vị trí trong mảng tăng dần."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Lấy từng phần tử chưa sắp xếp và chèn vào vị trí đúng trong phần đã được sắp xếp trước đó.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Tìm vị trí chèn bằng cách so sánh từ phải sang trái với các phần tử đã sắp xếp và lặp cho đến hết mảng.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 23,
                "snippet": "Bước 2: Lấy từng lá bài từ bàn và xem nó như một phần tử cần được sắp xếp.\nBước 3 : Chèn nó vào vị trí đúng trong tay trái",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 23,
                "snippet": "Để tìm vị trí đúng cho một lá bài, chúng ta so sánh nó với mỗi lá bài đã có trong tay trái, từ\nphải sang trái.\nLuôn luôn, các lá bài nằm trong tay trái đã được sắp xếp.\nBước 4 : Lặp lại các bước 2 đến 3 cho đến khi tất cả các lá bài được chèn vào tay.",
            },
        ],
    },
    {
        "query_id": "Q024",
        "query": "Giải thích hoạt động của hàm đệ quy BINARY-SEARCH(A, l, r, key) và điều kiện nào của dữ liệu đầu vào là bắt buộc để thuật toán này đúng?",
        "topic": "Searching & Sorting",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "pseudocode_interpretation",
        "reference_answer": (
            "Tìm kiếm nhị phân chỉ áp dụng trên mảng đã được sắp xếp. "
            "Trong BINARY-SEARCH(A, l, r, key): nếu l > r thì trả về -1 (không tìm thấy); ngược lại tính chỉ số giữa m = (l + r)/2, "
            "nếu A[m] = key thì trả về m, nếu A[m] > key thì gọi đệ quy trên nửa trái (l, m-1), ngược lại gọi đệ quy trên nửa phải (m+1, r)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Điều kiện bắt buộc: mảng đầu vào phải đã được sắp xếp.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Điều kiện dừng l > r trả về -1; so sánh key với phần tử giữa A[m] và đệ quy sang nửa trái (l, m-1) hoặc nửa phải (m+1, r).", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/013_train_file_Vie.pdf", "page": 4,
                "snippet": "- Sửdụng cho việc tìm kiếm trên mảng đã được sắp xếp",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/013_train_file_Vie.pdf", "page": 4,
                "snippet": "Function BINARY-SEARCH(A,l, r, key)\n\n                1. If (l> r) return -1;\n\n                2. m = (l+r) /2 ;\n\n                3. If (A[m] = key ) return m ;\n\n                4. Else if (A[m] > key) return BINARY-SEARCH(A, l, m-1, key);\n\n                5. Else return BINARY-SEARCH(A, m+1, r, key);",
            },
        ],
    },
    {
        "query_id": "Q025",
        "query": "Trong giải thuật Bubble Sort (Procedure BUBBLE-SORT), số phép so sánh và đổi chỗ ở trường hợp tốt nhất và xấu nhất là bao nhiêu?",
        "topic": "Searching & Sorting",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "time_complexity",
        "reference_answer": (
            "Ở trường hợp tốt nhất khi dãy ban đầu đã được sắp xếp, Procedure BUBBLE-SORT thực hiện 0 phép đổi chỗ nhưng vẫn thực hiện n(n-1)/2 phép so sánh. "
            "Ở trường hợp xấu nhất, giải thuật thực hiện n(n-1)/2 phép so sánh và n(n-1)/2 phép đổi chỗ; độ phức tạp thời gian trung bình là O(n^2)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Trường hợp tốt nhất (đã sắp xếp): 0 phép đổi chỗ và n(n-1)/2 phép so sánh.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Trường hợp xấu nhất: n(n-1)/2 phép đổi chỗ và so sánh; độ phức tạp trung bình O(n^2).", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/012_train_file_Vie.pdf", "page": 10,
                "snippet": "z Trường hợp tốt nhất:\n\n- Dãy ban đầu đã được sắp xếp\n\n                - 0 thực hiện phép đổi chỗ, n(n-1)/2 phép so sánh",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/012_train_file_Vie.pdf", "page": 10,
                "snippet": "z Trường hợp xấu nhất\n\n                - n(n-1)/2 phép đổi chỗvà so sánh\n\n                - Độphức tạp thời gian trung bình O(n2)",
            },
        ],
    },
    {
        "query_id": "Q026",
        "query": "So sánh tính ổn định (stability), bộ nhớ phụ và độ phức tạp trường hợp xấu nhất giữa Quick Sort và Merge Sort.",
        "topic": "Searching & Sorting",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "comparison",
        "reference_answer": (
            "Quick Sort không ổn định (có thể làm thay đổi thứ tự tương đối của các phần tử bằng nhau), hoạt động hiệu quả về bộ nhớ không cần mảng phụ lớn, nhưng trường hợp xấu nhất suy biến xuống O(n^2) (khi mảng đã sắp xếp hoặc đảo ngược). "
            "Ngược lại, Merge Sort là thuật toán ổn định, đảm bảo độ phức tạp O(n log n) ở cả trường hợp tốt nhất, trung bình và xấu nhất (nhưng cần mảng tạm temp khi trộn)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Quick Sort không ổn định, ít tốn bộ nhớ phụ nhưng xấu nhất O(n^2) khi mảng đã sắp xếp hoặc đảo ngược.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Merge Sort ổn định và giữ O(n log n) trong cả tốt nhất, trung bình và xấu nhất (dùng mảng tạm khi merge).", "evidence_mode": "composite"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 33,
                "snippet": "- Quick sort hoạt động hiệu quả trên bộ nhớ do không cần phải sử dụng thêm bộ nhớ\nphụ.\n3.4.5.3. Nhược điểm\n- Quick sort có thể trở nên chậm nếu mảng đã được sắp xếp hoặc đảo ngược.\n- Trong trường hợp xấu nhất, khi mảng đã được sắp xếp hoặc đảo ngược, quick sort có\n      thể làm việc với độ phức tạp O(n^2).\n- Quick sort không ổn định, nghĩa là vị trí tương đối của các phần tử bằng nhau có thể bị\nthay đổi sau khi được sắp xếp.",
            },
            {
                "ku_id": "KU2", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 36,
                "snippet": "3.5.5.1. Độ phức tạp\n   - Trung bình: O(nlogn)\n   - Tốt nhất: O(nlogn)\n   - Xấu nhất: O(nlogn)\n3.5.5.2. Ưu điểm\n- Merge sort là một thuật toán ổn định, nghĩa là nó không đổi vị trí các phần tử bằng\nnhau trong mảng đầu vào.",
            },
            {
                "ku_id": "KU2", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 35,
                "snippet": "int* temp = new int[right - left + 1];",
            },
        ],
    },
    {
        "query_id": "Q027",
        "query": "How does Radix Sort combine base-n tuple representation with Counting Sort, and why must the auxiliary Counting Sort be stable and process digits from least significant to most significant?",
        "topic": "Searching & Sorting",
        "language": "en",
        "difficulty": "medium",
        "question_type": "algorithm_steps",
        "reference_answer": (
            "Radix Sort represents each integer key (< n^c) as a c-digit tuple in base n (via repeated divmod by n) and sorts one digit position at a time using Counting Sort. "
            "Digits must be sorted from least significant to most significant, and Counting Sort must be stable (preserving insertion order within each chain) so that ties on higher-significance digits retain the sorted order established by previous passes."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Represent keys < n^c as c base-n digit tuples and sort each digit with Counting Sort.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Sort from least significant to most significant digit using a stable sort so earlier digit order is preserved.", "evidence_mode": "composite"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/005_train_file_Eng.pdf", "page": 4,
                "snippet": "If every key < nc for some positive c = logn(u), every key has at most c digits base n\n\n      - A c-digit number can be written as a c-element tuple in O(c) time\n\n      - We sort each of the c base-n digits in O(n) time",
            },
            {
                "ku_id": "KU2", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/005_train_file_Eng.pdf", "page": 4,
                "snippet": "- Sort least signiﬁcant key b, then most signiﬁcant key a\n\n- Stability ensures previous sorts stay sorted",
            },
            {
                "ku_id": "KU2", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/005_train_file_Eng.pdf", "page": 3,
                "snippet": "- Want sort to be stable: repeated keys appear in output in same order as input",
            },
        ],
    },
    {
        "query_id": "Q028",
        "query": "Why is the worst-case running time of any comparison-based sorting algorithm lower bounded by Omega(n log n)?",
        "topic": "Searching & Sorting",
        "language": "en",
        "difficulty": "medium",
        "question_type": "time_complexity",
        "reference_answer": (
            "In the comparison model, the algorithm's decision tree is binary (constant branching factor) and must have at least L >= n! leaves to distinguish all n! possible permutations of an n-element array. "
            "Therefore, the tree height — and thus worst-case running time — is lower bounded by Omega(log(n!)) >= log((n/2)^(n/2)) = Omega(n log n)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Comparison model induces a binary decision tree requiring at least L >= n! leaves for n! permutations.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Minimum height is Omega(log L) = Omega(log(n!)) = Omega(n log n).", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/005_train_file_Eng.pdf", "page": 2,
                "snippet": "- Comparison model implies that algorithm decision tree is binary (constant branching factor)\n\n- Requires # leaves L ≥ # possible outputs",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/005_train_file_Eng.pdf", "page": 2,
                "snippet": "- To sort array of n elements, # outputs is n! permutations\n\n      - Thus height lower bounded by log(n!) ≥ log((n/2)n/2) = Ω(n log n)",
            },
        ],
    },

    # =========================================================================
    # TOPIC 6: Time & Space Complexity (4: 2 VI, 2 EN)
    # =========================================================================
    {
        "query_id": "Q029",
        "query": "Phân biệt ba ký hiệu tiệm cận Big O, Big Omega và Big Theta trong đánh giá hiệu năng thuật toán, và quy tắc rút gọn hàm f(n) theo Big O là gì?",
        "topic": "Time & Space Complexity",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "concept_explanation",
        "reference_answer": (
            "Big O mô tả trường hợp xấu nhất (cận trên đảm bảo thuật toán không vượt quá), Big Ω mô tả trường hợp tốt nhất và Big Θ mô tả trường hợp trung bình. "
            "Khi hàm f(n) chứa hằng số và nhiều bậc khác nhau của n, quy tắc Big O bỏ hằng số và giữ lại bậc cao nhất đại diện (ví dụ O(n^2 + 2n + 3) rút gọn thành O(n^2))."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Big O là cận trên/trường hợp xấu nhất, Big Ω là tốt nhất và Big Θ là trung bình.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Bỏ hằng số và chọn số hạng bậc cao nhất làm đại diện (vd O(n^2 + 2n + 3) -> O(n^2)).", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/006_train_file_Vie.docx", "page": 2,
                "snippet": "Worst-case (Trường hợp xấu nhất - Big O): Thường được dùng nhất để đảm bảo thuật toán không bao giờ vượt qua mức thời gian/không gian này.\n\nBest-case (Tốt nhất - Big Ω) và Average-case (Trung bình - Big Θ).",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/019_train_file_Vie.pdf", "page": 5,
                "snippet": "Trong trường hợp hàm f(n) có chứa hằng số và\ncác bậc khác nhau của n thì ta chọn bậc cao\n         nhất để đại diện cho hàm f(n)\n\n         O(n2 +2n +3) được thay bằng O(n2)",
            },
        ],
    },
    {
        "query_id": "Q030",
        "query": "Độ phức tạp không gian đo lường đại lượng nào, và sự khác biệt về cấp phát bộ nhớ giữa Stack Memory và Heap Memory cùng nguyên nhân gây ra lỗi Stack Overflow khi đệ quy là gì?",
        "topic": "Time & Space Complexity",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "space_complexity",
        "reference_answer": (
            "Độ phức tạp không gian (Space Complexity) đánh giá lượng bộ nhớ bổ sung thuật toán cần dùng khi thực thi. "
            "Stack Memory được trình biên dịch cấp phát tự động, truy cập rất nhanh nhưng giới hạn dung lượng (dùng cho biến cục bộ/khung ngăn xếp); Heap Memory được cấp phát động bằng new và phải tự giải phóng bằng delete. "
            "Khi gọi đệ quy quá sâu (thiếu điều kiện dừng hoặc input quá lớn), các stack frame đầy bộ nhớ Stack sẽ gây ra lỗi Stack Overflow."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Space Complexity đo lượng bộ nhớ bổ sung cần dùng khi chạy thuật toán.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Stack cấp phát tự động, nhanh nhưng giới hạn kích thước; Heap cấp phát động (new) và cần giải phóng (delete).", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Mỗi lần gọi đệ quy đẩy 1 stack frame lên Call Stack; đệ quy quá sâu/thiếu base case làm đầy Stack gây Stack Overflow.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/006_train_file_Vie.docx", "page": 2,
                "snippet": "Độ phức tạp không gian (Space Complexity): Đánh giá lượng bộ nhớ bổ sung thuật toán cần sử dụng.",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/006_train_file_Vie.docx", "page": 3,
                "snippet": "Stack Memory: Được cấp phát tự động bởi trình biên dịch. Truy cập rất nhanh nhưng kích thước giới hạn. Phù hợp cho các biến cục bộ trong hàm.\n\nHeap Memory: Cấp phát động (Dynamic Allocation) trong quá trình chạy (Runtime) sử dụng từ khóa new (trong C++). Lập trình viên phải tự giải phóng bộ nhớ bằng từ khóa delete để tránh rò rỉ.",
            },
            {
                "ku_id": "KU3", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/006_train_file_Vie.docx", "page": 4,
                "snippet": "Mỗi lần hàm được gọi, một khung ngăn xếp (stack frame) chứa các biến cục bộ và địa chỉ trả về sẽ được đẩy vào Call Stack. Nếu đệ quy quá sâu (quên base case hoặc khối lượng dữ liệu quá lớn), bộ nhớ Stack sẽ bị đầy, gây ra lỗi Stack Overflow phá vỡ chương trình.",
            },
        ],
    },
    {
        "query_id": "Q031",
        "query": "What is the formal definition of amortized cost T(n) over a sequence of operations, and why does inserting into a Dynamic Array achieve Theta(1) amortized time even though individual reallocations cost Theta(n)?",
        "topic": "Time & Space Complexity",
        "language": "en",
        "difficulty": "medium",
        "question_type": "time_complexity",
        "reference_answer": (
            "An operation has amortized cost T(n) if any sequence of k operations costs at most k * T(n) in total. "
            "In a Dynamic Array, whenever the array fills up (r = 1), allocating Theta(n) extra space (e.g., to fill ratio 1/2) costs Theta(n) once, "
            "but guarantees Theta(n) subsequent insertions before the next reallocation, so Theta(n) operations take Theta(n) total time — i.e., Theta(1) amortized time per operation."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Operation has amortized cost T(n) if k operations cost at most k*T(n).", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Allocating Theta(n) extra space when full ensures Theta(n) insertions before the next resize, so Theta(n) operations cost Theta(n) total.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/002_train_file_Eng.pdf", "page": 4,
                "snippet": "- Operation has amortized cost T (n) if k operations cost at most ≤ kT (n)",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/002_train_file_Eng.pdf", "page": 3,
                "snippet": "- Whenever array is full (r = 1), allocate Θ(n) extra space at end to ﬁll ratio ri (e.g., 1/2)\n\n- Will have to insert Θ(n) items before the next reallocation\n\n- A single operation can take Θ(n) time for reallocation\n\n- However, any sequence of Θ(n) operations takes Θ(n) time",
            },
        ],
    },
    {
        "query_id": "Q032",
        "query": "How are complexity classes P, NP, NP-hard, and NP-complete formally defined, and how do polynomial-time reductions establish that a problem is NP-complete?",
        "topic": "Time & Space Complexity",
        "language": "en",
        "difficulty": "hard",
        "question_type": "space_complexity",
        "reference_answer": (
            "P is the set of decision problems solvable in polynomial time poly(n). NP is the set of decision problems with a polynomial-time verifier V that accepts (I, c) for some polynomial-length certificate c iff I is a YES instance (so P subseteq NP subseteq EXP). "
            "A reduction A -> B converts A into B so B is at least as hard as A; a problem is NP-hard if every problem in NP reduces to it in polynomial time, and NP-complete = NP intersect NP-hard."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "P: solvable in poly(n); NP: polynomial-time verifier V with polynomial-length certificate c.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "NP-hard: every problem in NP is polynomially reducible to it; NP-complete = NP intersect NP-hard.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/012_train_file_Eng.pdf", "page": 2,
                "snippet": "- P is the set of decision problems for which there is an algorithm A such that, for every input\n     I of size n, A on I runs in poly(n) time and solves I correctly\n\n- NP is the set of decision problems for which there is a veriﬁcation algorithm V that takes as\ninput an input I of the problem and a certiﬁcate bit string of length polynomial in the size\nof I",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/012_train_file_Eng.pdf", "page": 3,
                "snippet": "- Problem A is NP-hard if every problem in NP is polynomially reducible to A\n\n- i.e., A is at least as hard as (can be used to solve) every problem in NP (X ≤ A for X ∈ NP)\n\n- NP-complete = NP ∩ NP-hard",
            },
        ],
    },

    # =========================================================================
    # TOPIC 7: Hashing (4: 2 VI, 2 EN)
    # =========================================================================
    {
        "query_id": "Q033",
        "query": "Đụng độ (collision) trong bảng băm là gì và phương pháp nối kết trực tiếp (chaining) giải quyết đụng độ như thế nào?",
        "topic": "Hashing",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "definition",
        "reference_answer": (
            "Đụng độ (collision) là hiện tượng hai khóa khác nhau (key1 != key2) nhưng khi qua hàm băm lại cho ra cùng một địa chỉ (f(key1) = f(key2)). "
            "Phương pháp nối kết trực tiếp (Chaining) giải quyết đụng độ bằng cách gom các nút bị băm vào cùng một địa chỉ thành một danh sách liên kết tại chỉ số đó."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Đụng độ xảy ra khi hai khóa khác nhau cho cùng giá trị hàm băm.", "evidence_mode": "alternative"},
            {"ku_id": "KU2", "description": "Nối kết trực tiếp (Chaining) gom các phần tử trùng địa chỉ thành một danh sách liên kết tại vị trí đó.", "evidence_mode": "alternative"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "alternative", "alt_group": "Q033_KU1",
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 126,
                "snippet": "- Đụng độ là hiện tượng các khóa khác nhau nhưng khi băm cho ra cùng địa chỉ như\nnhau.\n   - Khi key1 # key2 mà f(key1) = f(key2), chúng ta nói nút có khóa key1 đụng độ với nút có\nkhóa key2.",
            },
            {
                "ku_id": "KU1", "mode": "alternative", "alt_group": "Q033_KU1",
                "source_path": "docx_Files/vie_docx/004_train_file_Vie.docx", "page": 2,
                "snippet": "Xử lý đụng độ (Collision): Khi hai Key cho ra cùng một index",
            },
            {
                "ku_id": "KU2", "mode": "alternative", "alt_group": "Q033_KU2",
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 126,
                "snippet": "*Hướng giải quyết: Các nút bị băm cùng địa chỉ (đụng độ/ xung đột) sẽ được gom thành\nmột danh sách liên kết",
            },
            {
                "ku_id": "KU2", "mode": "alternative", "alt_group": "Q033_KU2",
                "source_path": "docx_Files/vie_docx/004_train_file_Vie.docx", "page": 2,
                "snippet": "ta xử lý bằng Chaining (Danh sách liên kết tại index đó) hoặc Open Addressing (Tìm ô trống tiếp theo).",
            },
        ],
    },
    {
        "query_id": "Q034",
        "query": "Tại sao dò tuyến tính (Linear Probing) dễ gây phân bố dồn cụm, và dò bậc hai (Quadratic Probing) cùng băm kép (Double Hashing) khắc phục bằng công thức hàm băm lại nào?",
        "topic": "Hashing",
        "language": "vi",
        "difficulty": "medium",
        "question_type": "application_scenario",
        "reference_answer": (
            "Dò tuyến tính H(k, i) = (h'(k) + i) mod m dễ rải nút không đều và nối tiếp thành cụm khi đụng độ liên tiếp làm truy xuất chậm. "
            "Dò bậc hai dùng f_i(key) = (f(key) + i^2) % M (nên chọn M là số nguyên tố) giúp rải nút đều hơn, tuy nhiên vẫn có thể không duyệt hết mọi ô trống với một số giá trị M. "
            "Băm kép dùng h(k, i) = (h1(k) + i * h2(k)) mod m với h1 = k mod m và h2 = 1 + k mod m' (m' < m, m nguyên tố)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Dò tuyến tính H(k,i) = (h'(k)+i) mod m dễ phân bố không đều/nối tiếp nhau làm chậm truy xuất.", "evidence_mode": "composite"},
            {"ku_id": "KU2", "description": "Dò bậc hai dùng fi(key) = (f(key) + i^2) % M (nên chọn M nguyên tố, nhưng có thể không duyệt hết mọi ô trống).", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Băm kép dùng h(k,i) = (h1(k) + i*h2(k)) mod m với h1 = k mod m và h2 = 1 + k mod m'.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 130,
                "snippet": "- Hạn chế của bảng băm dùng phương pháp dò tuyến tính bị hạn chế do rải các nút\nkhông đều, nếu liên tục xảy ra đụng độ thì các khóa sẽ được phân bố nối tiếp nhau\ntheo thứ tự từ trên xuống. Dẫn đến việc truy xuất chậm.",
            },
            {
                "ku_id": "KU1", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/014_train_file_Vie.pdf", "page": 21,
                "snippet": "z H(k,i) = (h’(k) + i ) mod m",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 130,
                "snippet": "fi(key) = (f(key) + i2) % M với f(key) là hàm băm chính của bảng băm.\n- Khi sử dụng phương pháp băm bậc 2 nên chọn số địa chỉ M là số nguyên tố\n- Sẽ có trường hợp bảng băm còn trống vị trí để chứa khóa, nhưng khi băm sẽ không tìm\nra được vị trí để điền khóa vào bảng băm.",
            },
            {
                "ku_id": "KU3", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/014_train_file_Vie.pdf", "page": 22,
                "snippet": "h(k,i) = (h1(k) + i*h2(k)) mod m (i = 0, 1,…, m-1)\n\nz Thông thường m được chọn là sốnguyên tố\n\n                z h1 = k mod m ; h2 = 1+ k mod m’ với m’ < m và m’ rất gần với\n\nm",
            },
        ],
    },
    {
        "query_id": "Q035",
        "query": "Why must collisions inevitably occur when hashing a large key universe u into a table of size m << u, and why does a fixed division hash function h(k) = k mod m have a Theta(n) worst-case chain?",
        "topic": "Hashing",
        "language": "en",
        "difficulty": "easy",
        "question_type": "concept_explanation",
        "reference_answer": (
            "By the pigeonhole principle, when m << u, no hash function mapping {0,...,u-1} to {0,...,m-1} can be injective, so distinct keys a != b with h(a) = h(b) always exist. "
            "Furthermore, when u >> n, any fixed deterministic hash function such as division h(k) = k mod m has an adversarial input set of keys that all map to the same slot, creating a Theta(n)-length chain."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "By pigeonhole principle, m << u means no hash function is injective, so collisions always exist.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "For u >> n, any fixed hash function (like k mod m) has an input set producing an O(n)/Theta(n) chain.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/004_train_file_Eng.pdf", "page": 2,
                "snippet": "- If m ≪ u, no hash function is injective by pigeonhole principle",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/004_train_file_Eng.pdf", "page": 3,
                "snippet": "Division (bad):\n                h(k) = (k mod m)",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/004_train_file_Eng.pdf", "page": 3,
                "snippet": "If u ≫ n, every hash function will have some input set that will a create O(n) size chain",
            },
        ],
    },
    {
        "query_id": "Q036",
        "query": "Why is a Direct Access Array impractical in space when n << u, and how does Universal Hashing H(p, m) bound expected chain length to O(1) while using Theta(n) space?",
        "topic": "Hashing",
        "language": "en",
        "difficulty": "hard",
        "question_type": "space_complexity",
        "reference_answer": (
            "A Direct Access Array requires O(u) space (e.g., ~17.6 TB for 10-letter names), which is prohibitive when n << u. "
            "Universal Hashing chooses h_{ab}(k) = (((ak + b) mod p) mod m) randomly from H(p, m) with prime p > u and table size m = Theta(n), guaranteeing Pr[h(k_i) = h(k_j)] <= 1/m for k_i != k_j. "
            "By linearity of expectation, expected chain length is E[X_i] <= 1 + (n - 1)/m = O(1) since load factor alpha = n/m = O(1)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Direct Access Array uses O(u) space, which is prohibitive when n << u.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Universal family h_ab(k) = (((ak+b) mod p) mod m) satisfies Pr[h(k_i)=h(k_j)] <= 1/m.", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Expected chain size <= 1 + (n-1)/m = O(1) for m = Omega(n), giving expected amortized O(1) operations in Theta(n) space.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/004_train_file_Eng.pdf", "page": 2,
                "snippet": "But space O(u), so really bad if n ≪ u... :(\n\n- Example: if keys are ten-letter names, for one bit per name, requires 2610 ≈ 17.6 TB space",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/004_train_file_Eng.pdf", "page": 4,
                "snippet": "Universal (good, theoretically):\n                hab(k) = (((ak + b) mod p) mod m)\n\n   - Hash Family H(p, m) = {hab | a, b ∈{0, . . . , p − 1} and a 6= 0}\n\n   - Parameterized by a ﬁxed prime p > u, with a and b chosen from range {0, . . . , p − 1}\n\n   - H is a Universal family: Pr {h(ki) = h(kj )} ≤ 1/m",
            },
            {
                "ku_id": "KU3", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/004_train_file_Eng.pdf", "page": 4,
                "snippet": "Since m = Ω(n), load factor α = n/m = O(1), so O(1) in expectation!",
            },
        ],
    },

    # =========================================================================
    # TOPIC 8: Algorithm Design Techniques (4: 2 VI, 2 EN)
    # =========================================================================
    {
        "query_id": "Q037",
        "query": "Kỹ thuật Quy hoạch động bao lồi (Convex Hull Trick) giải quyết bài toán cực trị tập hàm bậc nhất như thế nào, giảm độ phức tạp từ bao nhiêu xuống bao nhiêu và áp dụng ra sao cho bài toán Commando (APIO 2010)?",
        "topic": "Algorithm Design Techniques",
        "language": "vi",
        "difficulty": "hard",
        "question_type": "multi_evidence_reasoning",
        "reference_answer": (
            "Kỹ thuật bao lồi (Convex Hull Trick) tiền xử lý tập các hàm bậc nhất y = a_i*x + b_i trên hệ tọa độ Oxy để tìm nhanh cực trị tại truy vấn x bằng cách loại bỏ các đường thẳng không bao giờ đạt cực trị, "
            "giảm độ phức tạp từ O(M*Q) xuống O((Q + M) log M) và tối ưu hóa các bài toán quy hoạch động từ O(n^2) xuống O(n log n) hoặc O(n). "
            "Trong bài toán Commando (APIO 2010), cần chia n binh sĩ thành các đoạn liên tiếp với hiệu quả điều chỉnh x' = ax^2 + bx + c (a < 0) sao cho tổng lớn nhất (ví dụ input 4, a=-1, b=10, c=-20, x=[2,2,3,4] cho kết quả tối ưu 9)."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Convex Hull Trick tìm cực trị tập hàm bậc nhất y=a_i*x+b_i bằng cách bỏ các đường không quan trọng.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Giảm độ phức tạp từ O(MQ) xuống O((Q+M)log M) và tối ưu QHĐ từ O(n^2) xuống O(n log n) hoặc O(n).", "evidence_mode": "composite"},
            {"ku_id": "KU3", "description": "Bài toán Commando (APIO 2010) tối đa hóa tổng hiệu quả điều chỉnh x'=ax^2+bx+c trên các đơn vị liên tiếp (ví dụ bảng Commando.inp -> 9).", "evidence_mode": "composite"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "docx_Files/vie_docx/002_train_file_Vie.docx", "page": 3,
                "snippet": "Kĩ thuật bao lồi\u00a0là kĩ thuật (hoặc là cấu trúc dữ liệu) dùng để xác định hiệu quả, có tiền xử lý, cực trị của một tập các hàm tuyến tính tại một giá trị của biến độc lập.",
            },
            {
                "ku_id": "KU2", "mode": "composite", "alt_group": None,
                "source_path": "docx_Files/vie_docx/002_train_file_Vie.docx", "page": 2,
                "snippet": "Kỹ thuật bao lồi (convex hull trick) dùng để tối ưu hóa thuật toán quy hoạch động với thời gian\u00a0O(n2)\u00a0xuống còn\u00a0O(nlogn), thậm chí với một số trường hợp xuống còn\u00a0O(n), cho một lớp các bài toán.",
            },
            {
                "ku_id": "KU2", "mode": "composite", "alt_group": None,
                "source_path": "docx_Files/vie_docx/002_train_file_Vie.docx", "page": 3,
                "snippet": "Nếu có\u00a0M đường thẳng và\u00a0Q\u00a0truy vấn, độ phức tạp của thuật toán sẽ\u00a0O(MQ). Kĩ thuật bao lồi sẽ giúp giảm độ phức tạp xuống còn\u00a0O((Q+M)logM",
            },
            {
                "ku_id": "KU3", "mode": "composite", "alt_group": None,
                "source_path": "docx_Files/vie_docx/002_train_file_Vie.docx", "page": 4,
                "snippet": "hiệu quả chiến đấu của một đơn vị nên được điều chỉnh như sau: điều chỉnh hiệu quả x’ được tính bằng cách sử dụng phương trình x’ = ax2 + bx + c, trong đó a, b, c là các hệ số đã biết (a <0)",
            },
            {
                "ku_id": "KU3", "mode": "composite", "alt_group": None,
                "source_path": "docx_Files/vie_docx/002_train_file_Vie.docx", "page": 4,
                "snippet": "| Commando.inp | Commando.out |\n| --- | --- |\n| 4 -1 10 -20 2 2 3 4 | 9 |",
            },
        ],
    },
    {
        "query_id": "Q038",
        "query": "Ý tưởng chia để trị (Divide and Conquer) trong Quick Sort thực hiện phân hoạch mảng xung quanh phần tử trục (pivot) như thế nào và điều kiện dừng đệ quy là gì?",
        "topic": "Algorithm Design Techniques",
        "language": "vi",
        "difficulty": "easy",
        "question_type": "application_scenario",
        "reference_answer": (
            "Quick Sort áp dụng chia để trị bằng cách chọn một phần tử làm trục (pivot) rồi phân hoạch mảng thành hai phần: các phần tử nhỏ hơn hoặc bằng pivot nằm phía trước, các phần tử lớn hơn pivot nằm phía sau, đặt pivot vào đúng vị trí giữa hai phần. "
            "Sau đó đệ quy áp dụng lên hai mảng con trái và phải; quá trình đệ quy kết thúc khi chỉ còn một phần tử hoặc không còn phần tử nào để phân chia."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Chọn pivot và phân hoạch thành hai phần (<= pivot bên trái, > pivot bên phải), đặt pivot đúng vị trí.", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Đệ quy hai phần trái/phải và dừng khi chỉ còn 1 phần tử hoặc mảng rỗng.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 31,
                "snippet": "Thuật toán Quick sort hoạt động bằng cách chọn một phần tử trong dãy đầu vào làm\n\"pivot\". Sau đó, các phần tử trong dãy đầu vào được phân thành hai phần: các phần tử\nnhỏ hơn hoặc bằng pivot được đưa về phía trước của pivot và các phần tử lớn hơn pivot\nđược đưa về phía sau của pivot.",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/vie_pdf/021_train_file_Vie.pdf", "page": 31,
                "snippet": "Sau đó, thuật toán sẽ đệ quy áp dụng phương pháp trên cho các phần tử nằm bên trái\nvà phải của pivot cho đến khi dãy được sắp xếp hoàn toàn. Quá trình đệ quy sẽ kết thúc\nkhi chỉ còn một phần tử hoặc không còn phần tử nào để phân chia.",
            },
        ],
    },
    {
        "query_id": "Q039",
        "query": "In the bottom-up dynamic programming algorithm lcs(A, B) for Longest Common Subsequence, what do the subproblems x[i][j], the recurrence relation, the base cases, and the loop iteration order represent?",
        "topic": "Algorithm Design Techniques",
        "language": "en",
        "difficulty": "medium",
        "question_type": "pseudocode_interpretation",
        "reference_answer": (
            "x[i][j] stores the length of the longest common subsequence of suffixes A[i:] and B[j:], with base cases x[i][|B|] = x[|A|][j] = 0 when either suffix is empty. "
            "If A[i] == B[j], x[i][j] = x[i+1][j+1] + 1; otherwise x[i][j] = max(x[i+1][j], x[i][j+1]). "
            "Because x[i][j] depends on strictly larger indices, the loops iterate in decreasing i and decreasing j, returning x[0][0] in O(|A| * |B|) time."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Subproblem x(i, j) = LCS length of suffixes A[i:] and B[j:], base x(i, |B|) = x(|A|, j) = 0.", "evidence_mode": "composite"},
            {"ku_id": "KU2", "description": "Recurrence: x[i+1][j+1]+1 if A[i]==B[j], else max(x[i+1][j], x[i][j+1]).", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Topological order decreases i and j; returns x[0][0] in O(|A|*|B|) time.", "evidence_mode": "composite"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/015_train_file_Eng.pdf", "page": 2,
                "snippet": "x(i, j) = length of longest common subsequence of sufﬁxes A[i :] and B[j :]",
            },
            {
                "ku_id": "KU1", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/015_train_file_Eng.pdf", "page": 2,
                "snippet": "x(i, |B|) = x(|A|, j) = 0 (one string is empty)",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/015_train_file_Eng.pdf", "page": 3,
                "snippet": "if A[i] == B[j]:\n\nx[i][j] = x[i + 1][j + 1] + 1\n               else:\n\nx[i][j] = max(x[i + 1][j], x[i][j\n       return x[0][0]\n\n+ 1])",
            },
            {
                "ku_id": "KU3", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/015_train_file_Eng.pdf", "page": 2,
                "snippet": "Nice order for bottom-up code: Decreasing i, then decreasing j",
            },
            {
                "ku_id": "KU3", "mode": "composite", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/015_train_file_Eng.pdf", "page": 3,
                "snippet": "- # subproblems: (|A| + 1) · (|B| + 1)\n\n          - work per subproblem: O(1)\n\n          - O(|A| · |B|) running time",
            },
        ],
    },
    {
        "query_id": "Q040",
        "query": "In Dynamic Programming for Longest Increasing Subsequence (LIS), why does simply defining subproblems on suffixes A[i:] fail, what extra constraint fixes it, and how are the recurrence and final answer formulated?",
        "topic": "Algorithm Design Techniques",
        "language": "en",
        "difficulty": "hard",
        "question_type": "multi_evidence_reasoning",
        "reference_answer": (
            "Defining subproblems only on suffixes A[i:] does not carry enough structure to enforce the strictly increasing condition when recursing on A[i+1:]. "
            "Adding the constraint that x(i) is the LIS length of suffix A[i:] that MUST include A[i] fixes this, yielding recurrence x(i) = max({1 + x(j) | i < j < |A|, A[j] > A[i]} union {1}) computed in decreasing i. "
            "Because the overall LIS can start at any index, the original problem answer is max{x(i) | 0 <= i < |A|}."
        ),
        "required_knowledge_units": [
            {"ku_id": "KU1", "description": "Unconstrained suffix subproblems cannot guarantee increasing order on A[i+1:]; fix by requiring A[i] to be included in x(i).", "evidence_mode": "single"},
            {"ku_id": "KU2", "description": "Recurrence x(i) = max({1 + x(j) | i < j < |A|, A[j] > A[i]} U {1}) solved in decreasing i.", "evidence_mode": "single"},
            {"ku_id": "KU3", "description": "Final answer guesses the first element: max{x(i) | 0 <= i < |A|}.", "evidence_mode": "single"},
        ],
        "evidences": [
            {
                "ku_id": "KU1", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/015_train_file_Eng.pdf", "page": 4,
                "snippet": "- But then how do we recurse on A[i + 1 :] and guarantee increasing subsequence?\n\n- Fix: add constraint to subproblems to give enough structure to achieve increasing\n\nproperty\n\n1. Subproblems\n\n       - x(i) = length of longest increasing subsequence of sufﬁx A[i :] that includes A[i]",
            },
            {
                "ku_id": "KU2", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/015_train_file_Eng.pdf", "page": 4,
                "snippet": "x(i) = max{1 + x(j) | i < j < |A|, A[j] > A[i]} [ {1}\n\n3. Topological order\n\n- Decreasing i",
            },
            {
                "ku_id": "KU3", "mode": "single", "alt_group": None,
                "source_path": "pdf_Files/eng_pdf/015_train_file_Eng.pdf", "page": 4,
                "snippet": "5. Original problem\n\n- What is the ﬁrst element of LIS? Guess!\n\n       - Length of LIS of A is max{x(i) | 0  i < |A|}",
            },
        ],
    },
]
