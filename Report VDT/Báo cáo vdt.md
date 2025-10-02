# BÁO CÁO ĐỀ TÀI

# **Hệ thống Trợ lý Y tế Đa phương thức sử dụng Multi-Agent** 

## **1\. Giới thiệu đề tài**

**Bối cảnh thực tiễn:**

Thực tiễn hiện nay cho thấy nhu cầu hỗ trợ về chăm sóc sức khỏe ban đầu và tiếp cận thông tin y tế của người dân ngày càng tăng cao. Tuy nhiên, quá trình này đang đối mặt với một số thách thức cụ thể như sau:

* Người dùng có nhu cầu được hỗ trợ nhanh chóng trong việc nhận diện, đánh giá các triệu chứng phổ biến như cảm cúm, dị ứng, rối loạn tiêu hóa… Tuy nhiên, việc tiếp cận các dịch vụ tư vấn y tế kịp thời và chính xác còn hạn chế, đặc biệt ở khu vực ngoài đô thị hoặc trong các tình huống khẩn cấp.  
* Nhận thức và kiến thức về chăm sóc sức khỏe tại nhà của phần lớn người dân còn hạn chế. Nhiều người chưa nắm vững các biện pháp xử lý ban đầu đối với các vấn đề sức khỏe thông thường, dẫn đến nguy cơ tự ý điều trị sai cách hoặc bỏ qua các dấu hiệu cảnh báo quan trọng.  
* Đa số người dùng không chuyên gặp khó khăn trong việc tiếp cận, tra cứu và hiểu các nguồn thông tin y khoa chính thống. Điều này làm gia tăng nguy cơ tiếp nhận thông tin sai lệch hoặc áp dụng các biện pháp chăm sóc sức khỏe không phù hợp, ảnh hưởng tiêu cực đến hiệu quả phòng bệnh và điều trị.

**Bối cảnh công nghệ:**  
Bên cạnh các thách thức thực tiễn, sự phát triển nhanh chóng của trí tuệ nhân tạo (AI) trong lĩnh vực y tế, đặc biệt là các hệ thống đa tác nhân (multi-agent) và AI đa phương thức (multimodal AI), đang mở ra nhiều cơ hội mới. Các mô hình này cho phép hệ thống hiểu, kết hợp và xử lý đồng thời nhiều dạng dữ liệu như văn bản, hình ảnh, âm thanh… Nhờ đó, khả năng nhận diện, phân tích triệu chứng và cung cấp thông tin y khoa ngày càng chính xác, cá nhân hóa và phù hợp với từng người dùng. Tuy nhiên, việc tích hợp hiệu quả các công nghệ này vào thực tiễn vẫn là một bài toán lớn, đòi hỏi giải pháp tổng thể và linh hoạt.

Tổng quan, bối cảnh thực tế đặt ra yêu cầu cấp thiết đối với việc phát triển các giải pháp hỗ trợ tiếp cận thông tin y tế chính xác, dễ hiểu cho mọi đối tượng người dùng. Việc giải quyết các thách thức này có ý nghĩa quan trọng trong việc nâng cao hiệu quả chăm sóc sức khỏe ban đầu, giảm tải cho hệ thống y tế và góp phần bảo vệ sức khỏe cộng đồng một cách bền vững

**Mục tiêu:**

Từ những phân tích trên, đề tài hướng tới việc xây dựng một hệ thống Multi-Agent trợ lý y tế ảo có khả năng phân tích triệu chứng đa phương thức và cung cấp thông tin y khoa đáng tin cậy cho người dùng phổ thông. 

Việc ứng dụng mô hình đa tác nhân kết hợp AI đa phương thức sẽ giúp hệ thống tiếp nhận, xử lý và giải thích nhiều loại dữ liệu khác nhau, từ đó hỗ trợ người dùng hiệu quả hơn trong việc nhận diện và xử lý các vấn đề sức khỏe thường gặp.

## **2\. Giải pháp đề xuất**

**2.1. Multi-Agent Medical Assistant** 

- Giải pháp đề xuất là một nền tảng ứng dụng trí tuệ nhân tạo dựa trên mô hình AI đa phương thức (Multimodal AI). Hệ thống này có khả năng hiểu, kết hợp và xử lý đồng thời nhiều dạng thông tin khác nhau, bao gồm văn bản, hình ảnh, và có thể mở rộng ra các dạng dữ liệu khác trong tương lai.  
- Multi-Agent Medical Assistant được thiết kế để cung cấp một trải nghiệm toàn diện và thông minh cho người dùng trong việc quản lý sức khỏe cá nhân. 

**2.2. Chức năng cơ bản**   
Hệ thống Multi-Agent Medical Assistant sẽ cung cấp các chức năng cơ bản sau: 

* Hỏi đáp sức khỏe thường gặp: Người dùng có thể đặt câu hỏi về các vấn đề sức khỏe phổ biến và nhận được câu trả lời chính xác, dễ hiểu. Chức năng này được hỗ trợ bởi các mô hình AI chuyên biệt cho từng lĩnh vực y tế.   
    
* Tư vấn theo triệu chứng đầu vào: Dựa trên các triệu chứng mà người dùng cung cấp (có thể là văn bản mô tả hoặc hình ảnh), hệ thống sẽ phân tích và đưa ra các gợi ý ban đầu về tình trạng sức khỏe. Để hỗ trợ chức năng này, hệ thống tích hợp các công cụ tìm kiếm kiến thức y khoa (web search) để truy xuất thông tin cập nhật và đáng tin cậy.   
    
* Tra cứu thông tin thuốc và bệnh lý: Người dùng có thể tra cứu thông tin chi tiết về các loại thuốc, bệnh lý, và các điều kiện y tế khác. Chức năng này được hỗ trợ bởi việc kết nối với các cơ sở dữ liệu bệnh, thuốc và triệu chứng chuyên sâu, đảm bảo thông tin cung cấp là chính xác và đầy đủ.

## 

## 

## **3\. Kiến trúc hệ thống**

**3.1. Kiến trúc tổng quan**

![][image1]  
Hệ thống gồm 2 Module chính:

**\- Vision Module**: Thành phần này chịu trách nhiệm xử lý và kết hợp các dạng thông tin hình ảnh. Nó có khả năng nhận diện loại ảnh, áp dụng các mô hình phù hợp để phân tích, và đưa ra kết quả phân tích. Thông tin phân tích từ Vision Agent cũng được lưu trữ trong bộ nhớ chia sẻ (shared memory) để phục vụ cho các lần hỏi sau, giúp hệ thống có thể duy trì ngữ cảnh và cải thiện độ chính xác của các phản hồi. 

**\- Text Module**: Thành phần này tập trung vào việc trích xuất các thông tin cần thiết từ các nguồn văn bản, tổng hợp và đưa ra kết quả. Text Agent đóng vai trò quan trọng trong việc xử lý các truy vấn dựa trên văn bản và cung cấp thông tin y khoa từ các nguồn đáng tin cậy.

**3.2. Kiến trúc chi tiết**

![][image2]

**Luồng hoạt động của tư vấn & trả lời:** 

Luồng hoạt động của quá trình tư vấn và trả lời trong hệ thống được thiết kế một cách có tổ chức để đảm bảo hiệu quả và chính xác. Các bước chính bao gồm:

1. **Tiếp nhận yêu cầu, phân tích và routing đến các tool agent**: Khi người dùng đưa ra yêu cầu, hệ thống sẽ tiếp nhận, phân tích nội dung yêu cầu và định tuyến đến các tác tử (agent) chuyên biệt phù hợp. Việc định tuyến này đảm bảo rằng yêu cầu được xử lý bởi thành phần có khả năng tốt nhất.   
2. **Xử lý yêu cầu**: Tùy thuộc vào loại yêu cầu, một hoặc nhiều tác tử sẽ được kích hoạt để xử lý:  
- **Vision Module**: Được sử dụng khi yêu cầu liên quan đến thông tin hình ảnh, như phân tích triệu chứng qua ảnh chụp.   
- **Rag Agent and Web Search Agent**: Được sử dụng để truy xuất thông tin từ cơ sở dữ liệu nội bộ (RAG Agent) hoặc tìm kiếm thông tin mới nhất từ internet (Web Search Agent) khi cần thiết.   
- **Conversational Agent**: Xử lý các cuộc hội thoại thông thường không yêu cầu phân tích chuyên sâu về hình ảnh hoặc truy xuất dữ liệu đặc biệt.   
3. **Kiểm tra kết quả phản hồi**: Sau khi các tác tử xử lý yêu cầu, hệ thống sẽ kiểm tra và đánh giá chất lượng của các phản hồi được tạo ra. Bước này đảm bảo rằng thông tin cung cấp là chính xác và phù hợp  
4. **Đưa ra kết quả cho người dùng**: Cuối cùng, hệ thống sẽ tổng hợp các phản hồi và trình bày kết quả cho người dùng một cách rõ ràng và dễ hiểu

**3.3. Decision Agent**

Decision Agent đóng vai trò là bộ điều phối trung tâm, sử dụng LangGraph để điều phối các tác tử chuyên biệt. Chức năng chính của nó là phân tích yêu cầu của người dùng và định tuyến yêu cầu đó tới tác tử thích hợp nhất để xử lý. Điều này đảm bảo rằng mỗi yêu cầu được xử lý bởi thành phần có chuyên môn cao nhất, tối ưu hóa hiệu suất và độ chính xác của hệ thống

**3.4. Vision Module**

![][image3]

**Vision Module** được thiết kế để xử lý các thông tin liên quan đến hình ảnh. Khi nhận được dữ liệu hình ảnh, Vision Agent sẽ nhận diện loại ảnh và áp dụng mô hình phù hợp để phân tích. Kết quả phân tích từ các mô hình này sau đó được đưa vào một bộ tổng hợp (summarizer) để tổng hợp thông tin và đưa ra một số gợi ý ban đầu cho người dùng. Đặc biệt, kết quả phân tích cũng được lưu trữ trong bộ nhớ chia sẻ (shared memory) để phục vụ cho các lần hỏi sau, giúp hệ thống duy trì ngữ cảnh và cải thiện khả năng phản hồi theo thời gian. 

***Chú thích mở rộng*****:**

Kiến trúc này cho phép dễ dàng tích hợp thêm các mô hình AI chuyên biệt cho từng nhóm bệnh hoặc loại dữ liệu y tế cụ thể, ví dụ như mô hình chẩn đoán ung thư, bệnh tim mạch, thần kinh, da liễu, hoặc các mô hình phân tích hình ảnh y học tiên tiến (MRI, CT, X-quang, v.v.). Nhờ đó, hệ thống có thể nâng cao độ chính xác trong nhận diện và hỗ trợ đa dạng các tình huống lâm sàng, đáp ứng tốt hơn nhu cầu thực tiễn và sự phát triển của công nghệ AI y tế hiện đại.

**3.5. Text Module**

![][image4]

**RAG Agent (Retrieval-Augmented Generation Agent)**

* RAG Agent chịu trách nhiệm truy xuất thông tin từ cơ sở dữ liệu vector (Qdrant).  
* Để nâng cao hiệu quả truy vấn, agent này thực hiện mở rộng truy vấn bằng cách bổ sung các thuật ngữ y khoa liên quan.  
* RAG Agent áp dụng kỹ thuật tìm kiếm kết hợp (hybrid search), tích hợp cả hai phương pháp: tìm kiếm từ khóa (BM25) và tìm kiếm bằng vector dày đặc (dense vector). Điều này giúp đảm bảo kết quả truy vấn có độ chính xác và mức độ liên quan cao.  
* Sau khi truy xuất, các kết quả sẽ được sắp xếp lại bằng Reranker. Quá trình này nhằm tối ưu hóa thứ tự hiển thị, ưu tiên các thông tin quan trọng nhất cho người dùng.

**Web Search Agent**

* Web Search Agent được kích hoạt trong hai trường hợp: khi kết quả từ RAG Agent có độ tin cậy thấp hoặc khi cần truy xuất các thông tin mới nhất.  
* Agent này thực hiện truy xuất thông tin từ các nguồn đáng tin cậy trên internet, điển hình như Tavily, PubMed và các nguồn y khoa uy tín khác.  
* Chức năng chính của Web Search Agent là tổng hợp, cập nhật thông tin mới nhất từ internet, bổ sung vào kho kiến thức của hệ thống.  
* Nhờ đó, hệ thống đảm bảo các phản hồi luôn được cập nhật, chính xác và phù hợp với thực tiễn mới nhất

**Normal Conversation Agent**

* Thành phần chịu trách nhiệm xử lý các cuộc hội thoại thông thường giữa người dùng và hệ thống.  
* Tập trung vào các tương tác tự nhiên như hỏi đáp, trò chuyện, hướng dẫn hoặc giải thích thông tin mà không cần phân tích chuyên sâu về hình ảnh hay truy xuất dữ liệu đặc biệt.

**3.6.  Kiểm soát độ chính xác và rủi ro của MLLM trong phân tích hình ảnh y tế**

## **4\. Triển khai công nghệ**

Để xây dựng hệ thống Multi-Agent trợ lý y tế ảo với khả năng phân tích triệu chứng đa phương thức và cung cấp thông tin y khoa đáng tin cậy, đề tài lựa chọn tích hợp các công nghệ hiện đại, tối ưu cho từng thành phần của workflow AI phức tạp. Việc kết hợp các nền tảng và mô hình dưới đây giúp đảm bảo hiệu quả, khả năng mở rộng và độ tin cậy của hệ thống: 

- **LangGraph**: Framework mã nguồn mở cho phép xây dựng và điều phối các workflow AI phức tạp theo mô hình đồ thị. LangGraph hỗ trợ quản lý trạng thái, điều hướng luồng xử lý, tích hợp nhiều tác nhân AI và kiểm soát linh hoạt quá trình xử lý, phù hợp với các ứng dụng đa tác nhân và workflow có nhiều bước rẽ nhánh, lặp lại.  
- **E5 Multilingual Large Instruct**: Mô hình embedding đa ngôn ngữ mạnh mẽ, hỗ trợ hơn 100 ngôn ngữ, được tối ưu cho các tác vụ tìm kiếm ngữ nghĩa và các bài toán nhúng văn bản phức tạp. Mô hình này tạo ra vector biểu diễn ngữ nghĩa chất lượng cao, phù hợp cho các hệ thống truy xuất thông tin đa ngôn ngữ, đặc biệt là tiếng Việt.  
- **Qdrant Vector Database**: Cơ sở dữ liệu vector chuyên dụng, tối ưu cho lưu trữ và tìm kiếm dữ liệu vector hiệu suất cao. Qdrant hỗ trợ tìm kiếm theo độ tương đồng ngữ nghĩa, tích hợp các thuật toán tìm kiếm gần đúng (ANN) và cho phép kết hợp truy vấn vector với các bộ lọc dữ liệu bổ sung, rất phù hợp cho ứng dụng AI cần truy xuất thông tin chính xác, nhanh chóng.  
- **Gemini 2.0 Flash**: Mô hình ngôn ngữ lớn (LLM) thế hệ mới, hỗ trợ xử lý đa phương thức (text, hình ảnh, âm thanh, video), tốc độ phản hồi nhanh, tối ưu cho các tác vụ tạo nội dung, tổng hợp thông tin và hội thoại chuyên sâu. Gemini 2.0 Flash nổi bật với khả năng reasoning minh bạch, tích hợp công cụ ngoài và hỗ trợ hơn 100 ngôn ngữ.  
- **Tavily Web Search**: Công cụ tìm kiếm web thời gian thực dành cho AI, cung cấp API cho phép truy xuất, tổng hợp thông tin mới nhất từ các nguồn uy tín trên internet. Tavily hỗ trợ tích hợp dễ dàng vào workflow của AI agent, đảm bảo dữ liệu luôn được cập nhật, xác thực và phù hợp với nhu cầu của hệ thống

## **5\. Dataset** 

Trong hệ thống Trợ lý Y tế Đa phương thức, bộ dữ liệu VimedAQA được sử dụng làm cơ sở dữ liệu chính cho tác nhân RAG. VimedAQA là bộ dữ liệu y tế tiếng Việt toàn diện, bao gồm nhiều lĩnh vực như bệnh tật, thuốc và các thông tin y tế liên quan khác. Việc sử dụng bộ dữ liệu chuyên biệt và đa dạng này giúp hệ thống có thể truy xuất, hiểu và xử lý các truy vấn y tế bằng tiếng Việt một cách chính xác, đồng thời cung cấp các phản hồi phù hợp với bối cảnh y tế tại Việt Nam.  
![][image5]

## **6\. Kết quả & Đánh giá**

Đề tài không sử dụng các kết quả benchmark để đánh giá mà chỉ tiến hành chạy thực nghiệm mô phỏng trên giao diện web demo: 

- Giao diện tương tác real-time demo hoạt động ổn định.  
- Quá trình điều phối, phân tích và sử dụng agent được hiển thị từng bước và cho phép người dùng nhìn thấy kết quả được suy luận từ các nguồn. 

**\- Ưu điểm:** 

- Tính mô-đun cao, dễ mở rộng.  
- Có khả năng tương tác với dữ liệu multimodal (text \+ ảnh).  
- Sử dụng cho domain đặc thù là y tế, có nhiều tiềm năng để phát triển  
- Tính thực tế cao với nhu cầu của người dùng, tăng cường hiểu biết và các cách xử lý ban đầu của người dùng với các loại bệnh.   
- Độ tin cậy cao khi trả lời dựa trên các tài liệu y tế chuyên khoa và các lời tư vấn của bác sĩ trong database hoặc các tổ chức uy tín trên internet. 

**\- Hạn chế:** 

- Dữ liệu y tế cho database còn hạn chế.  
- Cần thêm một số mô hình AI chuyên biệt để dự đoán các bệnh theo các chuyên ngành đặc biệt.  
- Cần sử dụng các mô hình LLM có kiến thức về y tế với các agent sử dụng LLM để chẩn đoán bệnh.  
- Cần sử dụng mô hình embedding tối ưu hơn với nhiều khả năng xử lý được long context tốt hơn, đặc biệt với chuyên ngành y tế có nhiều văn bản dài và phức tạp.

Nhìn chung, hướng tiếp cận này phù hợp với thực tiễn phát triển các hệ thống AI y tế hiện đại, tạo nền tảng vững chắc cho việc mở rộng chức năng, nâng cao chất lượng phục vụ và thích ứng linh hoạt với các yêu cầu mới trong lĩnh vực chăm sóc sức khỏe số.

## **7\. Tiến độ & Hướng phát triển**

**Tiến độ hiện tại:**

Hoàn thiện các module cốt lõi ở mức cơ bản:

* Đã xây dựng và triển khai Decision Agent với chức năng điều phối trung tâm, sử dụng LangGraph để định tuyến yêu cầu đến các agent phù hợp.  
* Vision Module đã hoàn thành ở mức cơ bản, hỗ trợ nhận diện và phân tích các thông tin hình ảnh, lưu trữ kết quả vào bộ nhớ chia sẻ để duy trì ngữ cảnh.  
* Text Module đã hoàn thiện:   
  * Tích hợp RAG Agent để truy xuất thông tin từ cơ sở dữ liệu y tế tiếng Việt (VimedAQA) và Web Search Agent để cập nhật thông tin từ các nguồn uy tín trên internet.  
  * Normal Conversation Agent đã được triển khai, đảm bảo khả năng xử lý các hội thoại thông thường với người dùng.

**Hướng phát triển** trong tương lai của đề tài tập trung vào những điểm sau:

* Tăng cường hợp tác và giao tiếp giữa các agent chuyên biệt nhằm nâng cao khả năng xử lý các truy vấn phức tạp, đa chiều và đa nguồn dữ liệu, giúp hệ thống phản ứng linh hoạt và chính xác hơn trong các tình huống đa dạng.

* Áp dụng cơ chế học thích nghi (adaptive learning) để hệ thống có thể tự động cải thiện chất lượng phản hồi dựa trên tương tác thực tế với người dùng, từ đó nâng cao độ chính xác và tính phù hợp của các kết quả theo thời gian.  
    
* Mở rộng khả năng tương tác đa phương thức (multimodal), tích hợp sâu hơn dữ liệu văn bản, hình ảnh, âm thanh và thiết bị y tế đeo tay, giúp hệ thống hỗ trợ chẩn đoán và theo dõi sức khỏe toàn diện hơn.

* Phát triển các mô hình AI chuyên biệt cho từng chuyên ngành y tế, như dự đoán bệnh lý chuyên sâu, hỗ trợ chẩn đoán phân biệt, và quản lý bệnh mãn tính cá nhân hóa, nhằm nâng cao giá trị ứng dụng trong thực tiễn

## **8\. Kết luận**

Dự án đã xây dựng thành công nền tảng hệ thống Trợ lý Y tế Đa phương thức dựa trên kiến trúc multi-agent, đáp ứng tốt các yêu cầu cơ bản về phân tích triệu chứng, truy xuất thông tin y tế và hỗ trợ tư vấn cho người dùng phổ thông tại Việt Nam. Hệ thống thể hiện tính mô-đun, khả năng mở rộng và tích hợp đa dạng công nghệ hiện đại, đồng thời đảm bảo độ tin cậy nhờ truy xuất từ các nguồn dữ liệu chuyên ngành và uy tín. Tuy còn một số hạn chế về quy mô dữ liệu và độ sâu chuyên môn của các mô hình AI, kết quả thực nghiệm cho thấy hướng tiếp cận này phù hợp với thực tiễn phát triển các hệ thống AI y tế hiện đại, tạo nền tảng vững chắc cho việc mở rộng chức năng, nâng cao chất lượng phục vụ và thích ứng linh hoạt với các yêu cầu mới trong lĩnh vực chăm sóc sức khỏe số. Các hướng phát triển tiếp theo sẽ tập trung vào tối ưu hóa workflow, mở rộng dữ liệu, phát triển các mô hình AI chuyên biệt và tăng cường khả năng tương tác đa phương thức, nhằm nâng cao hiệu quả và giá trị ứng dụng thực tiễn của hệ thống.

## **9\. References**

… (Thêm vào trong latex)

